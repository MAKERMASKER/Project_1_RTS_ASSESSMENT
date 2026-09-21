"""OutcomeExtractor: per-trial framework-vs-marker comparison metrics.

Ports resources/_old_reference_code/analysis.py's analyze_trial(), split from
LSI aggregation (see lsi.py) and rewritten for clarity.

Deliberately NOT ported: the old code's exo-torque mechanistic correction to
EXO-condition moments (adding the assistive device's simulated torque curve
back onto the measured moment). This project's isokinetic reference values
are raw measured torques, so framework/marker moments must stay raw too for
a like-for-like comparison against the isokinetic LSI -- an "exo-corrected"
moment would compare apples to oranges.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as _student_t

from config import DEFAULT
from discovery import TrialDiscovery
from insole_geometry import InsoleGeometry
from io_tables import read_storage, read_trc, segments

METRIC_COLUMNS = [
    "Trial", "Activity", "Condition", "Side",
    "PeakFrameworkGRF_N", "PeakMarkerGRF_N", "GrfRmse_N",
    "GrfXRmse_N", "GrfZRmse_N",
    "CopApRmse_mm", "CopMlRmse_mm", "Cop2dRmse_mm",
    "HipMomentRmse_Nm", "KneeMomentRmse_Nm", "AnkleMomentRmse_Nm",
    "PeakFrameworkHip_Nm", "PeakFrameworkKnee_Nm", "PeakFrameworkAnkle_Nm",
    "PeakMarkerHip_Nm", "PeakMarkerKnee_Nm", "PeakMarkerAnkle_Nm",
    "SupportMomentRmse_Nm", "PeakFrameworkSupportMoment_Nm", "PeakMarkerSupportMoment_Nm",
    "FrameworkJumpHeight_m", "MarkerJumpHeight_m",
    "JumpDistance_m", "SuccessfulLateralJumps",
    "FrameworkHipNetWork_J", "FrameworkKneeNetWork_J", "FrameworkAnkleNetWork_J",
    "MarkerHipNetWork_J", "MarkerKneeNetWork_J", "MarkerAnkleNetWork_J",
    "FrameworkHipAbsoluteWork_J", "FrameworkKneeAbsoluteWork_J", "FrameworkAnkleAbsoluteWork_J",
    "MarkerHipAbsoluteWork_J", "MarkerKneeAbsoluteWork_J", "MarkerAnkleAbsoluteWork_J",
    "FrameworkTotalNetWork_J", "MarkerTotalNetWork_J",
    "FrameworkTotalAbsoluteWork_J", "MarkerTotalAbsoluteWork_J",
]


def _rms(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values ** 2))) if len(values) else np.nan


def _interp(source_time, values, target_time):
    return np.interp(target_time, source_time, values, left=np.nan, right=np.nan)


def _guarded(mask, time, guard_s: float = 0.03) -> np.ndarray:
    """Erode a boolean mask so isolated True samples near a False boundary drop out."""
    guard = int(np.ceil(guard_s / np.median(np.diff(time))))
    mask = np.asarray(mask, bool)
    out = np.zeros_like(mask)
    for index in range(len(mask)):
        out[index] = mask[max(0, index - guard):min(len(mask), index + guard + 1)].all()
    return out


def _flight_time_height(time, contact, landing_peak_index: int, gravity_mps2: float) -> float:
    """Traditional jump height: g * flight_time^2 / 8, independent for each contact signal."""
    contact = np.asarray(contact, bool)
    landing = segments(contact)
    landing = landing[(landing[:, 0] <= landing_peak_index) & (landing[:, 1] >= landing_peak_index)]
    if not len(landing):
        return np.nan
    flights = segments(~contact)
    flights = flights[flights[:, 1] < landing[0, 0]]
    if not len(flights):
        return np.nan
    duration = float(time[int(landing[0, 0])] - time[int(flights[-1, 0])])
    return gravity_mps2 * duration ** 2 / 8


class OutcomeExtractor:
    def __init__(self, config=DEFAULT):
        self.config = config
        self.discovery = TrialDiscovery(config)

    def trial_summary_path(self, subject, activity) -> Path:
        return self.config.results_root / subject.id / "analysis" / activity.name / "trial_summary.csv"

    def run(self, subject, activity, make_plots: bool = True) -> pd.DataFrame:
        rows = []
        series_list = []
        for condition in self.config.conditions:
            for trial in self.discovery.trials(subject, activity, condition, for_analysis=True):
                result = self.analyze_trial(subject, activity, trial, make_plots)
                if result is not None:
                    row, series = result
                    rows.append(row)
                    series_list.append(series)
        if not rows:
            raise RuntimeError(f"No analyzable {activity.name} trials for {subject.id}")
        summary = pd.DataFrame(rows, columns=METRIC_COLUMNS)
        out = self.trial_summary_path(subject, activity)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out, index=False)
        if make_plots and series_list:
            self._plot_combined_grf_ci(subject, activity, series_list)
        return summary

    def analyze_trial(self, subject, activity, trial, make_plot: bool = True):
        package = trial.output / "trial_data.npz"
        framework_id = trial.output / f"{trial.name}_opencap_insole_id.sto"
        marker_root = self.config.results_root / subject.id / "marker_based" / trial.condition
        marker_ik_path = marker_root / "ik" / f"{trial.name}_ik.mot"
        marker_id_path = marker_root / "id" / f"{trial.name}_inverse_dynamics.sto"
        required = (package, framework_id, marker_ik_path, marker_id_path, trial.force_mot)
        if not all(path.is_file() for path in required):
            return None

        data = np.load(package, allow_pickle=True)
        if bool(data.get("quality_omit", False)):
            return None

        time = data["time"]
        force = data["force"]
        contact = data["contact"].astype(bool)
        grf = data["grf_N"]
        cop = data["cop_m"]

        framework_ik = pd.DataFrame(data["ik_values"], columns=[str(name) for name in data["ik_names"]])
        framework_id_table = read_storage(framework_id)
        marker_id_table = read_storage(marker_id_path)
        marker_ik_table = read_storage(marker_ik_path)
        force_plate = read_storage(trial.force_mot)
        plate = 1 if trial.side == "R" else 2

        marker_vertical = np.abs(force_plate[f"ground_force_{plate}_vy"].to_numpy(float))
        shift = float(time[np.argmax(force)] - force_plate.time.iloc[np.argmax(marker_vertical)])
        marker_force = _interp(force_plate.time + shift, marker_vertical, time)
        marker_force_x = _interp(
            force_plate.time + shift, force_plate[f"ground_force_{plate}_vx"].to_numpy(float), time
        )
        marker_force_z = _interp(
            force_plate.time + shift, force_plate[f"ground_force_{plate}_vz"].to_numpy(float), time
        )

        marker_cop = force_plate[[f"ground_force_{plate}_p{axis}" for axis in "xyz"]].to_numpy(float)
        if np.nanmedian(np.abs(marker_cop)) > 2:
            marker_cop = marker_cop / 1000
        marker_local = InsoleGeometry.to_local(
            read_trc(trial.marker_trc), force_plate.time.to_numpy(), marker_cop, trial.side
        )
        marker_local = np.column_stack([
            _interp(force_plate.time + shift, marker_local[:, axis], time) for axis in range(2)
        ])
        framework_local = InsoleGeometry.to_local(
            read_trc(trial.effective_opencap_trc()), time, cop, trial.side
        )[:, :2]

        shared = _guarded(contact & (marker_force >= self.config.contact_force_n), time)
        marker_local = marker_local.copy()
        framework_local = framework_local.copy()
        marker_local[~shared] = np.nan
        framework_local[~shared] = np.nan

        suffix = trial.side.lower()
        angles = [f"hip_flexion_{suffix}", f"knee_angle_{suffix}", f"ankle_angle_{suffix}"]
        moments = [f"{angle}_moment" for angle in angles]
        framework_angle = np.column_stack([framework_ik[angle] for angle in angles])
        marker_angle = np.column_stack([
            _interp(marker_ik_table.time + shift, marker_ik_table[angle], time) for angle in angles
        ])
        framework_moment = np.column_stack([
            np.interp(time, framework_id_table.time, framework_id_table[moment]) for moment in moments
        ])
        marker_moment = np.column_stack([
            _interp(marker_id_table.time + shift, marker_id_table[moment], time) for moment in moments
        ])

        peak_index = int(np.argmax(grf[:, 1]))
        marker_contact_segments = segments(contact & (marker_force >= self.config.contact_force_n))
        landing_segment = marker_contact_segments[
            (marker_contact_segments[:, 0] <= peak_index) & (marker_contact_segments[:, 1] >= peak_index)
        ]
        if not len(landing_segment):
            return None
        landing = np.zeros(len(time), bool)
        landing[landing_segment[0, 0]:landing_segment[0, 1] + 1] = True

        framework_peak = np.nanmax(np.abs(framework_moment[landing]), axis=0)
        marker_peak = np.nanmax(np.abs(marker_moment[landing]), axis=0)
        moment_rmse = np.array([_rms(framework_moment[:, i] - marker_moment[:, i]) for i in range(3)])
        position_error = framework_local - marker_local

        # Winter's "support moment": the algebraic sum of the hip/knee/ankle
        # extensor moments, a single clinically-meaningful whole-limb number.
        # OpenSim reports these as flexion-coordinate moments (opposite sign
        # to an extensor moment), hence the leading minus sign.
        framework_support = -np.sum(framework_moment, axis=1)
        marker_support = -np.sum(marker_moment, axis=1)
        support_rmse = _rms(framework_support[landing] - marker_support[landing])
        framework_support_peak = float(np.nanmax(framework_support[landing]))
        marker_support_peak = float(np.nanmax(marker_support[landing]))

        framework_power = framework_moment * np.gradient(np.deg2rad(framework_angle), time, axis=0)
        marker_power = marker_moment * np.gradient(np.deg2rad(marker_angle), time, axis=0)
        framework_net_work = np.array([np.trapezoid(framework_power[landing, i], time[landing]) for i in range(3)])
        marker_net_work = np.array([np.trapezoid(marker_power[landing, i], time[landing]) for i in range(3)])
        framework_absolute_work = np.array([
            np.trapezoid(np.abs(framework_power[landing, i]), time[landing]) for i in range(3)
        ])
        marker_absolute_work = np.array([
            np.trapezoid(np.abs(marker_power[landing, i]), time[landing]) for i in range(3)
        ])

        marker_contact = np.isfinite(marker_force) & (marker_force >= self.config.contact_force_n)
        framework_jump_height = _flight_time_height(time, contact, peak_index, self.config.gravity_mps2)
        marker_jump_height = _flight_time_height(time, marker_contact, peak_index, self.config.gravity_mps2)
        jump_distance = float(data.get("jump_distance_m", np.nan))
        successful_jumps = int(data.get("successful_lateral_jumps", 0))

        row = [
            trial.name, activity.name, trial.condition, trial.side,
            np.nanmax(grf[landing, 1]), np.nanmax(marker_force[landing]),
            _rms(grf[shared, 1] - marker_force[shared]),
            _rms(grf[shared, 0] - marker_force_x[shared]), _rms(grf[shared, 2] - marker_force_z[shared]),
            1000 * _rms(position_error[:, 0]), 1000 * _rms(position_error[:, 1]),
            1000 * _rms(np.linalg.norm(position_error, axis=1)),
            *moment_rmse, *framework_peak, *marker_peak,
            support_rmse, framework_support_peak, marker_support_peak,
            framework_jump_height, marker_jump_height, jump_distance, successful_jumps,
            *framework_net_work, *marker_net_work,
            *framework_absolute_work, *marker_absolute_work,
            float(np.sum(framework_net_work)), float(np.sum(marker_net_work)),
            float(np.sum(framework_absolute_work)), float(np.sum(marker_absolute_work)),
        ]
        peak = float(data["peak_grf_time_s"])
        if make_plot:
            self._plot_trial(
                subject, trial, time, peak, grf,
                marker_force, marker_force_x, marker_force_z,
                framework_local, marker_local, framework_angle, marker_angle,
                framework_moment, marker_moment, framework_support, marker_support,
            )
        series = {
            "trial": trial.name,
            "x": time - peak,
            "framework": np.column_stack([grf[:, 0], grf[:, 1], grf[:, 2]]),
            "marker": np.column_stack([marker_force_x, marker_force, marker_force_z]),
        }
        return row, series

    def _plot_trial(self, subject, trial, time, peak, grf,
                     marker_force, marker_force_x, marker_force_z,
                     framework_cop, marker_cop, framework_angle, marker_angle,
                     framework_moment, marker_moment, framework_support, marker_support):
        """The main visual output for one trial: a 4x3 grid of framework (red)
        vs. marker/ground-truth (black) traces, each panel's title carrying
        the RMS (or peak, for support moment) disagreement between the two.
        Row 1: GRF Fx/Fy/Fz. Row 2: AP/ML COP, support moment. Row 3: hip/
        knee/ankle angle. Row 4: hip/knee/ankle moment."""
        out = self.config.results_root / subject.id / "analysis" / trial.activity / "trial_comparisons"
        out.mkdir(parents=True, exist_ok=True)
        x = time - peak

        figure, axes = plt.subplots(4, 3, figsize=(15, 15))
        axes = axes.ravel()

        axes[0].plot(x, grf[:, 0], "r", x, marker_force_x, "k")
        axes[0].set_title(f"Fx (AP) GRF | RMS {_rms(grf[:, 0] - marker_force_x):.1f} N")
        axes[1].plot(x, grf[:, 1], "r", x, marker_force, "k")
        axes[1].set_title(f"Fy (vertical) GRF | RMS {_rms(grf[:, 1] - marker_force):.1f} N")
        axes[2].plot(x, grf[:, 2], "r", x, marker_force_z, "k")
        axes[2].set_title(f"Fz (ML) GRF | RMS {_rms(grf[:, 2] - marker_force_z):.1f} N")

        axes[3].plot(x, framework_cop[:, 0], "r", x, marker_cop[:, 0], "k")
        axes[3].set_title(f"AP COP | RMS {1000 * _rms(framework_cop[:, 0] - marker_cop[:, 0]):.1f} mm")
        axes[4].plot(x, framework_cop[:, 1], "r", x, marker_cop[:, 1], "k")
        axes[4].set_title(f"ML COP | RMS {1000 * _rms(framework_cop[:, 1] - marker_cop[:, 1]):.1f} mm")
        axes[5].plot(x, framework_support, "r", label="Framework")
        axes[5].plot(x, marker_support, "k", label="Marker")
        axes[5].set_title(f"Support moment | RMS {_rms(framework_support - marker_support):.1f} Nm")
        axes[5].legend(fontsize=8)

        for index, label in enumerate(("Hip", "Knee", "Ankle")):
            axes[6 + index].plot(x, framework_angle[:, index], "r", x, marker_angle[:, index], "k")
            axes[6 + index].set_title(f"{label} angle")
            axes[9 + index].plot(x, framework_moment[:, index], "r", x, marker_moment[:, index], "k")
            axes[9 + index].set_title(
                f"{label} moment | RMS {_rms(framework_moment[:, index] - marker_moment[:, index]):.1f} Nm"
            )

        for axis in axes:
            axis.axvline(0, color="gray", linestyle="--", linewidth=0.8)
            axis.grid(True, alpha=0.3)

        figure.suptitle(f"{trial.name}  (red = framework, black = marker/ground truth)")
        figure.tight_layout()
        figure.savefig(out / f"{trial.name}_sync_grf_cop_angles_id.png", dpi=160)
        plt.close(figure)

    def _plot_combined_grf_ci(self, subject, activity, series_list, grid_half_width_s: float = 0.75, grid_dt_s: float = 0.01):
        """One combined plot, across every one of this subject's analyzable
        trials, of mean framework vs. marker GRF (Fx/Fy/Fz) with a 95% CI
        band at each timepoint -- a visual complement to the per-trial
        `_plot_trial` panels above, for seeing at a glance whether framework
        tracks marker within a consistent band or diverges systematically
        (e.g. a directional bias, as opposed to per-trial noise).

        Each trial's time axis is already `time - peak_grf_time` (seconds
        relative to that trial's own landing-GRF peak, matching
        `_plot_trial`'s x-axis) -- trials are combined by interpolating onto
        one common grid on that same axis, not by resampling to 0-100% of a
        variable-duration window, so "t=0" keeps the same physical meaning
        (the landing peak) across every trial.
        """
        grid = np.arange(-grid_half_width_s, grid_half_width_s + 1e-9, grid_dt_s)
        n_trials = len(series_list)
        framework_stack = np.full((n_trials, len(grid), 3), np.nan)
        marker_stack = np.full((n_trials, len(grid), 3), np.nan)
        for row, series in enumerate(series_list):
            for axis in range(3):
                framework_stack[row, :, axis] = np.interp(
                    grid, series["x"], series["framework"][:, axis], left=np.nan, right=np.nan
                )
                marker_stack[row, :, axis] = np.interp(
                    grid, series["x"], series["marker"][:, axis], left=np.nan, right=np.nan
                )

        def mean_and_ci(stack):
            # Trial windows vary slightly in length, so the grid's extremes
            # can have few or zero contributing trials -- nanmean/nanstd on
            # an empty slice is expected there (yields NaN, which
            # fill_between simply skips), not a real problem to warn about.
            count = np.sum(np.isfinite(stack), axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean = np.nanmean(stack, axis=0)
                sd = np.nanstd(stack, axis=0, ddof=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                standard_error = np.where(count > 1, sd / np.sqrt(np.maximum(count, 1)), np.nan)
                critical = _student_t.ppf(0.975, np.maximum(count - 1, 1))
            return mean, critical * standard_error, count

        framework_mean, framework_ci, trial_count = mean_and_ci(framework_stack)
        marker_mean, marker_ci, _ = mean_and_ci(marker_stack)

        out = self.config.results_root / subject.id / "analysis" / activity.name / "trial_comparisons"
        out.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(1, 3, figsize=(15, 5))
        for axis, label in enumerate(("Fx (AP)", "Fy (vertical)", "Fz (ML)")):
            ax = axes[axis]
            ax.plot(grid, framework_mean[:, axis], "r", label="Framework mean")
            ax.fill_between(
                grid, framework_mean[:, axis] - framework_ci[:, axis], framework_mean[:, axis] + framework_ci[:, axis],
                color="r", alpha=0.2, label="Framework 95% CI",
            )
            ax.plot(grid, marker_mean[:, axis], "k", label="Marker mean")
            ax.fill_between(
                grid, marker_mean[:, axis] - marker_ci[:, axis], marker_mean[:, axis] + marker_ci[:, axis],
                color="k", alpha=0.15, label="Marker 95% CI",
            )
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.grid(True, alpha=0.3)
            ax.set_title(label)
            ax.set_xlabel("Time relative to landing peak (s)")
            ax.set_ylabel("Force (N)")
        axes[0].legend(fontsize=8)
        figure.suptitle(
            f"{subject.id} | combined GRF, mean ± 95% CI across {n_trials} trials "
            "(red = framework, black = marker/ground truth)"
        )
        figure.tight_layout()
        figure.savefig(out / f"{subject.id}_combined_grf_ci.png", dpi=160)
        plt.close(figure)
