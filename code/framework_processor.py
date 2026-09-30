"""FrameworkProcessor: OpenCap + insole -> bilateral GRF/COP -> framework ID.

Written once; never changes when a new activity is added -- every
activity-specific decision (synchronization, whether the framework is
unilateral, extra outcomes) comes from the Activity object passed in.

GRF-from-markerless accuracy has a published, known limitation (see
resources/literature_notes.md): peak landing force from COM
double-differentiation is commonly biased low. The insole_vertical_weight /
com_vertical_weight blend below is a deliberate mitigation for that, not an
arbitrary constant -- its weight is an investigation target (see the plan's
"GRF accuracy investigation" step), not a fixed default.

Horizontal GRF (Fx/Fz) has its own, separate mitigation: pure horizontal COM
double-differentiation is known to be especially noisy from video-based
markerless kinematics. `rotation_force_weight_x`/`rotation_force_weight_z` in
subject_rules blend the translational estimate with a moment-balance
("H-dot") estimate -- the horizontal force implied by the body's own angular
momentum rate about the observed COP, via
`opensim_env.center_of_mass_and_momentum_rate()`. Gated per-activity by
`rotation_force_activities`, since it's only meaningful when
`activity.unilateral_framework` (a single, known point of force application).

Both of the above are differentiated (COM twice, momentum once) over the
trial's analysis window. `np.gradient` uses accurate central differences in
the interior of an array but falls back to a lower-order, one-sided
difference at the very first and last sample -- and double-differentiation
compounds that boundary error badly, producing a spurious spike right at the
edges of the window (most visible in Fx/Fz, since Fy for a subject with
insole_vertical_weight=1.0 doesn't use this term at all). `_padded_time()`
computes over a window extended by `config.com_gradient_pad_s` on each side
and crops back to the reported window afterward, so that boundary error
lands in the discarded padding instead of the output.

If a trial's TRC has been repaired (trc_repair.py), OpenCap's own cloud IK
reflects the ORIGINAL, unrepaired TRC and is stale -- `process()` reruns IK
from the repaired TRC via `opencap_ik.OpenCapIkRunner` (the same recipe
trc_repair.py's TrcRepairValidator uses for its own comparison) and caches
the result next to the repaired TRC, so it's computed once, not once per run.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calibration import InsoleCalibration
from config import DEFAULT
from discovery import TrialDiscovery
from insole_geometry import InsoleGeometry
from io_tables import interpolate_table, read_insole, read_storage, read_trc, write_bilateral_grf
from opencap_ik import OpenCapIkRunner
from opensim_env import center_of_mass, center_of_mass_and_momentum_rate
from scale_ik_id import InverseDynamicsRunner


class FrameworkProcessor:
    def __init__(self, config=DEFAULT):
        self.config = config
        self.discovery = TrialDiscovery(config)
        self.calibration = InsoleCalibration(config)
        self.id_runner = InverseDynamicsRunner(config)
        self.opencap_ik_runner = OpenCapIkRunner(config)

    def run(self, subject, activity, condition: str = "all", trial_name: str = None):
        completed = []
        for trial in self.discovery.trials(subject, activity, condition):
            if trial_name and trial.name.lower() != trial_name.lower():
                continue
            print(f"{trial.name}: bilateral GRF/COP -> ID")
            try:
                completed.append(self.process(subject, activity, trial))
            except FileNotFoundError as error:
                # A known-missing input (e.g. no OpenCap trial for this name --
                # see Preflight's MISSING_OPENCAP_TRC/MISSING_MATCHING_INSOLE_FILE)
                # skips this one trial; it must not abort the whole batch.
                print(f"  skipped ({error})")
            except ValueError as error:
                # A genuine per-trial data condition raised by
                # activity.synchronize() (e.g. no time overlap between the
                # insole and IK/TRC windows) -- same skip-not-abort handling
                # as a missing file, since it is specific to this one trial.
                print(f"  skipped ({error})")
        if not completed:
            raise RuntimeError(f"No matching {subject.id} trials were found.")
        return completed

    def process(self, subject, activity, trial) -> Path:
        opencap_trc = trial.effective_opencap_trc()
        opencap_ik_path = trial.effective_opencap_ik()
        for path in (opencap_trc, trial.insole):
            if not path or not Path(path).is_file():
                raise FileNotFoundError(path)

        # OpenCap's own scaled model for this session, untouched -- kept
        # deliberately independent of the marker-based pipeline's own
        # anatomical scaling (ModelScaler.scale()) so a framework-vs-marker
        # discrepancy can never be an artifact of sharing one model between
        # two pipelines that are supposed to be independent measurements.
        model_path = trial.opencap_model
        repaired = trial.repaired_opencap_trc is not None and trial.repaired_opencap_trc.is_file()
        if repaired and (self.config.overwrite or not opencap_ik_path.is_file()):
            print(f"  {trial.name}: rerunning OpenCap IK from the repaired TRC")
            self.opencap_ik_runner.run(
                opencap_trc, trial.opencap_model, opencap_ik_path,
                setup_folder=opencap_ik_path.parent / "setup",
            )
        if not opencap_ik_path.is_file():
            raise FileNotFoundError(opencap_ik_path)

        trc = read_trc(opencap_trc)
        ik_all = read_storage(opencap_ik_path)
        insole_all = read_insole(trial.insole)
        rules = subject.rules
        time, insole, ik, sync = activity.synchronize(trial, trc, ik_all, insole_all, rules)
        calibration = self.calibration.load(subject)
        geometry = InsoleGeometry.for_subject(subject)

        padded_time, core_start, core_stop = self._padded_time(time, ik_all)
        padded_ik = interpolate_table(ik_all, padded_time)

        com_padded, mass = center_of_mass(model_path, padded_ik)
        com_padded = self._smooth_com(com_padded, padded_time, self.config.com_smoothing_alpha)
        pad_dt = float(np.median(np.diff(padded_time)))
        acceleration_padded = np.column_stack([
            np.gradient(np.gradient(com_padded[:, axis], pad_dt), pad_dt) for axis in range(3)
        ])
        total_grf_padded = mass * (acceleration_padded - np.array([0, -self.config.gravity_mps2, 0]))
        com = com_padded[core_start:core_stop]
        total_grf = total_grf_padded[core_start:core_stop]

        rotation_com = momentum_rate = None
        if activity.name in rules.get("rotation_force_activities", []):
            rotation_com_padded, momentum_rate_padded = center_of_mass_and_momentum_rate(
                model_path, ik_all, padded_time, self.config.id_lowpass_hz
            )
            rotation_com = rotation_com_padded[core_start:core_stop]
            momentum_rate = momentum_rate_padded[core_start:core_stop]

        (right_force, right_cop, right_contact,
         left_force, left_cop, left_contact) = self._bilateral_forces(
            activity, trial, trc, geometry, time, insole, total_grf, calibration, rules,
            rotation_com, momentum_rate,
        )

        trial.output.mkdir(parents=True, exist_ok=True)
        grf_path = trial.output / f"{trial.name}_opencap_insole_grf.mot"
        write_bilateral_grf(grf_path, time, right_force, right_cop, left_force, left_cop)

        id_path = trial.output / f"{trial.name}_opencap_insole_id.sto"
        trial_data_path = trial.output / "trial_data.npz"
        interrupted_after_id = id_path.is_file() and not trial_data_path.is_file()
        if interrupted_after_id:
            print(f"  reusing completed ID after interrupted finalization: {trial.name}")
        else:
            self.id_runner.run_with_paths(
                model_path, opencap_ik_path, grf_path, id_path, time[0], time[-1],
                setup_folder=trial.output / "setup",
            )

        # sync["sync_side"] defaults to trial.side for every existing
        # activity (unchanged behavior) but lets a bilateral activity with
        # no single trial-wide side (SkaterHop: trial.side is None) report
        # which side its sync actually anchored on instead of silently
        # falling through to "left" via `None == "R"` -> False.
        selected_right = sync["sync_side"] == "R"
        selected_force = right_force if selected_right else left_force
        selected_cop = right_cop if selected_right else left_cop
        selected_contact = right_contact if selected_right else left_contact
        jump_distance, successful_jumps = activity.outcomes(trial, trc, sync)

        np.savez_compressed(
            trial_data_path,
            time=time, force=selected_force[:, 1], contact=selected_contact,
            grf_N=selected_force, cop_m=selected_cop,
            right_grf_N=right_force, right_cop_m=right_cop, right_contact=right_contact,
            left_grf_N=left_force, left_cop_m=left_cop, left_contact=left_contact,
            com_m=com, mass_kg=mass,
            processing_model=str(model_path),
            horizontal_force_method=("COM_Hdot_weighted" if momentum_rate is not None else "COM_translation"),
            rotation_force_weight_x=float(rules.get("rotation_force_weight_x", 0.0)),
            rotation_force_weight_z=float(rules.get("rotation_force_weight_z", 0.0)),
            momentum_rate_Nm=(momentum_rate if momentum_rate is not None else np.empty((0, 3))),
            ik_names=np.array(ik.columns), ik_values=ik.to_numpy(),
            shift_s=sync["shift_s"], peak_grf_time_s=sync["peak_grf_time_s"],
            method=sync["method"], quality_omit=sync["quality_omit"],
            jump_distance_m=jump_distance, successful_lateral_jumps=successful_jumps,
        )
        raw_selected_force = activity._selected_force(trial, insole)
        self._plot_sync(trial, trc, time, raw_selected_force, selected_force, sync)
        return trial_data_path

    # ---- internals --------------------------------------------------

    @staticmethod
    def _smooth_com(values: np.ndarray, time: np.ndarray, smoothing_alpha: float) -> np.ndarray:
        """Least-squares COM smoothing (translated from the study's MATLAB formulation)."""
        dt = np.median(np.diff(time))
        count = len(time)
        derivative = np.zeros((count - 4, count))
        for row in range(count - 4):
            derivative[row, row:row + 5] = [-0.5, 1, 0, -1, 0.5]
        derivative /= dt ** 3
        matrix = np.vstack([np.eye(count), smoothing_alpha * dt ** 3 * derivative])
        zeros = np.zeros(count - 4)
        return np.column_stack([
            np.linalg.lstsq(matrix, np.r_[values[:, axis], zeros], rcond=None)[0]
            for axis in range(3)
        ])

    def _padded_time(self, time: np.ndarray, ik_all):
        """Extend `time` by config.com_gradient_pad_s on each side, clipped to
        what `ik_all` actually covers, so that np.gradient's one-sided
        boundary difference -- badly compounded by double-differentiation --
        lands in the discarded padding rather than the reported window.

        Built by concatenation at the exact same sample spacing as `time`
        (not np.arange over the padded range), so `padded[core_start:
        core_stop]` recovers `time` exactly with no floating-point
        misalignment risk.
        """
        dt = float(np.median(np.diff(time)))
        pad_samples = max(1, int(round(self.config.com_gradient_pad_s / dt)))
        pre = time[0] - dt * np.arange(pad_samples, 0, -1)
        post = time[-1] + dt * np.arange(1, pad_samples + 1)
        available_min = float(ik_all.time.min())
        available_max = float(ik_all.time.max())
        pre = pre[pre >= available_min]
        post = post[post <= available_max]
        padded = np.concatenate([pre, time, post])
        core_start = len(pre)
        core_stop = core_start + len(time)
        return padded, core_start, core_stop

    @staticmethod
    def _vertical_scale(calibration: dict, activity, side: str) -> float:
        """This activity's own insole gain if the calibration carries one,
        else the subject-wide gain.

        The subject-wide value is fit from SLJ quiet stance only
        (calibration.py), so a calibration MAY additionally supply
        `per_activity[<activity>][<side>]["vertical_force_scale"]`. Whether
        those values are trustworthy is a property of the data written into
        the file (each carries its own `vertical_force_scale_source`), not of
        this lookup -- see expt_temp/cross_activity_gain_check/README.md.
        """
        override = (
            calibration.get("per_activity", {}).get(activity.name, {}).get(side, {})
        )
        if "vertical_force_scale" in override:
            return float(override["vertical_force_scale"])
        return float(calibration[side]["vertical_force_scale"])

    @staticmethod
    def _placement_for(calibration: dict, trial, side: str) -> dict:
        placement = dict(calibration[side])
        placement.update(calibration.get("trial_offsets", {}).get(trial.name, {}))
        return placement

    def _bilateral_forces(self, activity, trial, trc, geometry: InsoleGeometry,
                           time, insole, total_grf, calibration: dict, rules: dict,
                           rotation_com=None, momentum_rate=None):
        right_vertical = insole["right_force"].to_numpy(float, copy=True)
        left_vertical = insole["left_force"].to_numpy(float, copy=True)

        if activity.unilateral_framework:
            if trial.side == "R":
                left_vertical[:] = 0
            else:
                right_vertical[:] = 0

        right_contact = activity.contact(right_vertical)
        left_contact = activity.contact(left_vertical)
        right_vertical[~right_contact] = 0
        left_vertical[~left_contact] = 0

        total_vertical = right_vertical + left_vertical
        right_fraction = np.divide(
            right_vertical, total_vertical, out=np.zeros_like(total_vertical), where=total_vertical > 0
        )
        left_fraction = np.divide(
            left_vertical, total_vertical, out=np.zeros_like(total_vertical), where=total_vertical > 0
        )

        # Per-side calibrated gain -- calibration["R"] and calibration["L"] are
        # fit independently (see calibration.py); this is where that
        # correction actually takes effect.
        insole_weight = float(rules["insole_vertical_weight"])
        com_weight = float(rules["com_vertical_weight"])
        right_scale = self._vertical_scale(calibration, activity, "R")
        left_scale = self._vertical_scale(calibration, activity, "L")
        right_vertical = insole_weight * right_scale * right_vertical + com_weight * total_grf[:, 1] * right_fraction
        left_vertical = insole_weight * left_scale * left_vertical + com_weight * total_grf[:, 1] * left_fraction

        right_force = np.zeros((len(time), 3))
        left_force = np.zeros((len(time), 3))
        right_force[:, [0, 2]] = total_grf[:, [0, 2]] * right_fraction[:, None]
        left_force[:, [0, 2]] = total_grf[:, [0, 2]] * left_fraction[:, None]
        right_force[:, 1] = right_vertical
        left_force[:, 1] = left_vertical

        right_cop = geometry.transform(trc, time, insole, "R", self._placement_for(calibration, trial, "R"))
        left_cop = geometry.transform(trc, time, insole, "L", self._placement_for(calibration, trial, "L"))

        rotation_activities = rules.get("rotation_force_activities", [])
        weight_x = float(rules.get("rotation_force_weight_x", 0.0))
        weight_z = float(rules.get("rotation_force_weight_z", 0.0))
        if (activity.name in rotation_activities and activity.unilateral_framework
                and rotation_com is not None and momentum_rate is not None):
            # Blend the translational horizontal-force estimate with a
            # moment-balance estimate: the horizontal force implied by the
            # body's own angular-momentum rate about the observed COP,
            # treating the stance foot as the single point of force
            # application (only valid where lever[:,1] -- the vertical
            # distance from COM to COP -- is large enough to invert safely).
            force = right_force if trial.side == "R" else left_force
            cop = right_cop if trial.side == "R" else left_cop
            contact = right_contact if trial.side == "R" else left_contact
            lever = cop - rotation_com
            safe = contact & np.isfinite(lever).all(axis=1) & (np.abs(lever[:, 1]) > 0.25)
            rotation_x = np.zeros(len(time))
            rotation_z = np.zeros(len(time))
            rotation_x[safe] = (lever[safe, 0] * force[safe, 1] - momentum_rate[safe, 2]) / lever[safe, 1]
            rotation_z[safe] = (momentum_rate[safe, 0] + lever[safe, 2] * force[safe, 1]) / lever[safe, 1]
            force[safe, 0] = (1 - weight_x) * force[safe, 0] + weight_x * rotation_x[safe]
            force[safe, 2] = (1 - weight_z) * force[safe, 2] + weight_z * rotation_z[safe]

        right_bad = ~np.isfinite(right_cop).all(axis=1)
        left_bad = ~np.isfinite(left_cop).all(axis=1)
        right_force[~right_contact | right_bad] = 0
        left_force[~left_contact | left_bad] = 0
        right_cop[~right_contact | right_bad] = 0
        left_cop[~left_contact | left_bad] = 0
        return right_force, right_cop, right_contact, left_force, left_cop, left_contact

    def _plot_sync(self, trial, trc, time, raw_insole_force, framework_force, sync):
        """Match the toe-height/insole-force synchronization diagnostic plot.

        Uses sync["sync_side"], not trial.side directly -- for every
        existing activity these are the same (see Activity._finish_sync()),
        but SkaterHop's trial.side is None (bilateral, no single side) and
        InsoleGeometry.foot_height() would raise KeyError on a bare None.
        """
        relative_time = time - sync["peak_grf_time_s"]
        toe_height = np.interp(time, trc["time"], InsoleGeometry.foot_height(trc, sync["sync_side"]))

        figure, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
        force_axis = axes[0].twinx()
        axes[0].plot(relative_time, toe_height, color="blue")
        force_axis.plot(relative_time, raw_insole_force, color="red")
        axes[0].set_ylabel("Toe height (m)", color="blue")
        force_axis.set_ylabel("Insole force (N)", color="red")
        axes[0].set_title("Synchronization")

        axes[1].plot(relative_time, framework_force[:, 1], color="red", linewidth=1.2)
        axes[1].set_ylabel("Vertical GRF (N)")
        axes[1].set_xlabel("Time from landing GRF peak (s)")

        for axis in axes:
            axis.axvline(0, color="black", linestyle="--")
            axis.grid(True)
        axes[1].set_xlim(relative_time[0], relative_time[-1])
        figure.suptitle(trial.name)
        figure.tight_layout()
        figure.savefig(trial.output / "sync.png", dpi=150)
        plt.close(figure)
