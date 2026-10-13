"""collect_insole_com_rot_fy_visualization.py -- per-trial comparison of
RAW insole Fy, force-plate (ground truth) Fy, and a COM-acceleration +
H-dot (angular-momentum-rate) based Fy estimate from OpenCap kinematics
alone, all on the trial's own committed side (unilateral SLJ).

Sister to validation/sensor_insole/RAW_visualization/ (the plain
raw-insole-vs-plate script, ../expt_temp/collect_insole_raw_visualization.py),
output to a NEW sister folder:
    validation/sensor_insole/_RAWInsole_COM_ROT_Fplate_Fy/<SUBJECT>/<TRIAL>.png

Reuses the REAL production synchronization/calibration path (`activity
.synchronize()`, `InsoleCalibration.load()`, `InsoleGeometry.transform()`,
`FrameworkProcessor._padded_time`/`_smooth_com`) instead of re-deriving a
shift by hand, so this is on the exact same clock/scale convention as
`framework_processor.py`'s real output and `outcomes.py`'s own
marker-vs-framework alignment (`shift = time[argmax(framework)] -
force_plate.time[argmax(marker)]`, the same trick used here for the
force-plate overlay).

The "COM+ROT" Fy line, per explicit request (0.5/0.5 blend, computed for
ALL subjects/trials regardless of any subject_rules `rotation_force_*`
config, which only ever covers Fx/Fz in production):
    Fy_com_rot = 0.5 * Fy_translational + 0.5 * Fy_rotation
- Fy_translational = mass * (COM vertical acceleration + gravity), the
  padded/smoothed double-differentiation `framework_processor.py` already
  uses for `total_grf`.
- Fy_rotation is a NEW, NOT-PREVIOUSLY-USED-IN-PRODUCTION derivation.
  Production's existing H-dot correction (`_bilateral_forces`) solves the
  moment-balance equations for Fx/Fz TREATING Fy AS ALREADY KNOWN (the
  blended insole+COM estimate) -- by construction there is no "spare"
  equation left to solve for Fy the same way. This reuses the SAME M_z
  equation production uses for its `rotation_x` (Fx-solving) term, but
  solves it for Fy instead, treating the pure-translational Fx estimate
  as the known quantity (exactly what `force[:, 0]` holds at the point
  production computes `rotation_x`, before that update -- for a
  unilateral trial `right_fraction`/`left_fraction` is 1.0 on the trial's
  own side, so this is a direct match, not an approximation):
      M_z = lever_x*Fy - lever_y*Fx  =>  Fy = (M_z + lever_y*Fx) / lever_x
  Numerically riskier than production's Fx/Fz derivation: THAT divides by
  the vertical (large, ~0.3-1m) COM-to-COP lever arm; THIS divides by the
  anteroposterior (small, typically a few cm during single-leg stance)
  lever arm -- expect this term to be noisy. That is exactly what this
  diagnostic plot exists to reveal, not a bug to hide.

Run from code/, in the opensimPy environment:
    python expt_temp/collect_insole_com_rot_fy_visualization.py
    python expt_temp/collect_insole_com_rot_fy_visualization.py --subject AB08
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from activities import activity_for  # noqa: E402
from calibration import InsoleCalibration  # noqa: E402
from config import DEFAULT  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from framework_processor import FrameworkProcessor  # noqa: E402
from insole_geometry import InsoleGeometry  # noqa: E402
from io_tables import interpolate_table, read_insole, read_storage, read_trc  # noqa: E402
from opensim_env import center_of_mass, center_of_mass_and_momentum_rate  # noqa: E402
from subject import Subject  # noqa: E402

SUBJECT_IDS = ("AB03", "AB05", "AB07", "AB08", "AB10", "NTS01")
ACTIVITY_NAME = "single_leg_jump"
VALIDATION_ROOT = CODE_ROOT.parent / "validation" / "sensor_insole" / "_RAWInsole_COM_ROT_Fplate_Fy"

COLOR_PLATE = "purple"
COLOR_INSOLE = "deepskyblue"
COLOR_COM_ROT = "green"

# Safety floor on the anteroposterior COM-to-COP lever arm the Fy
# rotation term divides by -- small on purpose (this lever is
# fundamentally small, unlike the vertical one production's Fx/Fz
# derivation safely divides by), just enough to avoid a literal
# divide-by-near-zero.
MIN_AP_LEVER_M = 0.02


def _com_and_rotation(model_path, ik_all, time, fp: FrameworkProcessor):
    """(translational_grf, rotation_com, momentum_rate) on `time`, exactly
    mirroring FrameworkProcessor.process()'s own computation."""
    padded_time, core_start, core_stop = fp._padded_time(time, ik_all)
    padded_ik = interpolate_table(ik_all, padded_time)
    com_padded, mass = center_of_mass(model_path, padded_ik)
    com_padded = fp._smooth_com(com_padded, padded_time, DEFAULT.com_smoothing_alpha)
    pad_dt = float(np.median(np.diff(padded_time)))
    acceleration_padded = np.column_stack([
        np.gradient(np.gradient(com_padded[:, axis], pad_dt), pad_dt) for axis in range(3)
    ])
    translational_grf_padded = mass * (acceleration_padded - np.array([0, -DEFAULT.gravity_mps2, 0]))
    translational_grf = translational_grf_padded[core_start:core_stop]

    rotation_com_padded, momentum_rate_padded = center_of_mass_and_momentum_rate(
        model_path, ik_all, padded_time, DEFAULT.id_lowpass_hz
    )
    rotation_com = rotation_com_padded[core_start:core_stop]
    momentum_rate = momentum_rate_padded[core_start:core_stop]
    return translational_grf, rotation_com, momentum_rate


def _com_rot_fy(trial, activity, geometry, calibration, trc, time, insole, translational_grf, rotation_com, momentum_rate):
    """Fy_com_rot on `time`, for the trial's own committed side, zeroed
    outside of ground contact -- matching production's own convention
    (`_bilateral_forces()`'s `right_vertical[~right_contact] = 0` /
    final `right_force[~right_contact | right_bad] = 0`), using the
    same `activity.contact()` definition (raw insole force >= threshold)
    on the trial's own committed side."""
    side = trial.side
    own_column = "right_force" if side == "R" else "left_force"
    own_contact = activity.contact(insole[own_column].to_numpy(float))

    placement = FrameworkProcessor._placement_for(calibration, trial, side)
    cop = geometry.transform(trc, time, insole, side, placement)
    lever = cop - rotation_com

    fx_translational = translational_grf[:, 0]
    fy_translational = translational_grf[:, 1]
    m_z = momentum_rate[:, 2]

    safe = own_contact & np.isfinite(lever).all(axis=1) & (np.abs(lever[:, 0]) > MIN_AP_LEVER_M)
    fy_rotation = np.full(len(time), np.nan)
    fy_rotation[safe] = (m_z[safe] + lever[safe, 1] * fx_translational[safe]) / lever[safe, 0]

    fy_com_rot = np.where(np.isfinite(fy_rotation), 0.5 * fy_translational + 0.5 * fy_rotation, fy_translational)
    fy_com_rot = np.where(own_contact, fy_com_rot, 0.0)
    return fy_com_rot


def _plot_trial(trial, subject, activity, fp, calibration_obj, out_dir):
    trc = read_trc(trial.effective_opencap_trc())
    ik_all = read_storage(trial.effective_opencap_ik())
    insole_all = read_insole(trial.insole)
    rules = subject.rules
    time, insole, ik, sync = activity.synchronize(trial, trc, ik_all, insole_all, rules)

    calibration = calibration_obj.load(subject)
    geometry = InsoleGeometry.for_subject(subject)

    translational_grf, rotation_com, momentum_rate = _com_and_rotation(trial.opencap_model, ik_all, time, fp)
    com_rot_fy = _com_rot_fy(
        trial, activity, geometry, calibration, trc, time, insole, translational_grf, rotation_com, momentum_rate
    )

    own_column = "right_force" if trial.side == "R" else "left_force"
    own_insole = insole[own_column].to_numpy(float)

    force_plate = read_storage(trial.force_mot)
    own_plate_number = 1 if trial.side == "R" else 2
    plate_raw = np.abs(force_plate[f"ground_force_{own_plate_number}_vy"].to_numpy(float))
    shift = float(time[np.argmax(own_insole)] - force_plate.time.iloc[np.argmax(plate_raw)])
    own_plate = np.interp(time, force_plate.time + shift, plate_raw, left=np.nan, right=np.nan)

    figure, ax = plt.subplots(figsize=(11, 6))
    ax.plot(time, own_plate, color=COLOR_PLATE, label=f"Force plate {trial.side} (ground truth)", linewidth=1.4)
    ax.plot(time, own_insole, color=COLOR_INSOLE, label=f"Insole {trial.side} (raw)", linewidth=1.1)
    ax.plot(time, np.abs(com_rot_fy), color=COLOR_COM_ROT,
            label="COM + H-dot Fy (0.5/0.5, OpenCap-only)", linewidth=1.4, linestyle="--")
    ax.set_xlabel("Time (s, framework/OpenCap clock)")
    ax.set_ylabel("Force (N)")
    ax.set_title(f"{trial.name} (side {trial.side}) -- RAW insole vs. force plate vs. COM+H-dot Fy (shift={shift:.3f}s)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    figure.tight_layout()
    image_path = out_dir / f"{trial.name}.png"
    figure.savefig(image_path, dpi=150)
    plt.close(figure)
    return image_path


def collect(subject_id):
    subject = Subject(subject_id, DEFAULT)
    activity = activity_for(ACTIVITY_NAME, DEFAULT)
    discovery = TrialDiscovery(DEFAULT)
    fp = FrameworkProcessor(DEFAULT)
    calibration_obj = InsoleCalibration(DEFAULT)

    if not (DEFAULT.calibration_root / f"{subject_id}_insole_marker_foot_calibration.json").is_file():
        print(f"{subject_id}: no calibration file -- skipping")
        return

    trials = [
        trial
        for trial in discovery.trials(subject, activity, "all", for_analysis=True)
        if trial.insole.is_file() and trial.force_mot.is_file()
        and trial.effective_opencap_trc() and Path(trial.effective_opencap_trc()).is_file()
        and trial.effective_opencap_ik() and Path(trial.effective_opencap_ik()).is_file()
    ]
    if not trials:
        print(f"{subject_id}: no eligible trials, skipping")
        return

    out_dir = VALIDATION_ROOT / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)

    made = 0
    for trial in sorted(trials, key=lambda t: t.name):
        try:
            _plot_trial(trial, subject, activity, fp, calibration_obj, out_dir)
            made += 1
        except Exception as error:  # noqa: BLE001 -- report and keep going per trial
            print(f"  {trial.name}: FAILED ({type(error).__name__}: {error})")
    print(f"{subject_id}: {made}/{len(trials)} trials -> {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", help="Just this one subject instead of all 6")
    args = parser.parse_args()
    subject_ids = (args.subject,) if args.subject else SUBJECT_IDS
    for subject_id in subject_ids:
        collect(subject_id)


if __name__ == "__main__":
    main()
