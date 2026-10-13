"""collect_metrics.py -- per-trial kinematic predictors + ground-truth
outcomes for "can this trial support an Fy (vertical) gain estimate?".

Question this answers
---------------------
The COM+H-dot Fy estimator (see ../collect_insole_com_rot_fy_visualization.py)
works on some trials and is unusable on others. Outside the lab there is no
force plate to tell which is which, so we need a screening rule computable
from OpenCap kinematics + insole COP ALONE. This script computes candidate
predictors (ground-truth-free) AND the real outcome (ground-truth-based) for
every trial, so `analyze_screening.py` can test which predictors actually
discriminate -- i.e. use lab ground truth to validate a rule that will not
need ground truth.

Why insole COP is legitimate here: COP is a pressure *centroid*, invariant to
the overall force gain being calibrated. Scaling every pressure cell by a
constant leaves the centroid unchanged, so using it introduces no circularity.
(The per-subject rigid COP offset from `InsoleCalibration` is a geometric
constant, not a force-gain quantity -- but it IS a lab-fit value, which is a
real caveat for true field deployment; noted in README.md.)

The mechanism being screened for
--------------------------------
Sagittal-plane moment balance, the equation the Fy term inverts:
    M_z = lever_x * F_y - lever_y * F_x
    =>  F_y = (M_z + lever_y * F_x) / lever_x
`lever = cop - com`, so `lever_x` is the anteroposterior COM-to-COP offset.
Error in M_z is amplified by 1/|lever_x|; error in F_x (the noisiest channel)
by |lever_y|/|lever_x|. A trial only carries information about F_y when the
`lever_x * F_y` term is a substantial SHARE of the sagittal moment rather than
a small residual between two large terms.

The outcome variable, and the trap it avoids
--------------------------------------------
A trial can post a low whole-trace RMSE for the wrong reason: where the lever
is unsafe the blend falls back to pure COM-translational, which at rest simply
equals body weight and matches the plate. That is the H-dot term being
BYPASSED, not working. So the headline outcome here is
`delta_rmse_active_N` = RMSE(translational) - RMSE(blend), computed ONLY over
frames where the H-dot term is actually active. Positive means the H-dot term
genuinely improved on pure translational; negative means it made things worse.

Writes one row per trial to results/trial_metrics.csv, plus a per-trial npz
cache (results/cache/) so the analysis can be re-run without redoing the
expensive OpenSim COM/momentum-rate computation.

Run from code/:
    ..\\opensimPy\\Scripts\\python.exe expt_temp/fy_gain_trial_screening/collect_metrics.py
    ..\\opensimPy\\Scripts\\python.exe expt_temp/fy_gain_trial_screening/collect_metrics.py --subject AB05
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent.parent.parent
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
RESULTS = Path(__file__).resolve().parent / "results"

# Same floor the visualization script uses, kept here so the "active" mask
# this analysis reports on is exactly the one that produced those plots.
MIN_AP_LEVER_M = 0.02


def _com_and_rotation(model_path, ik_all, time, fp):
    """(translational_grf, rotation_com, momentum_rate) on `time` -- mirrors
    FrameworkProcessor.process()'s own computation, padding included."""
    padded_time, core_start, core_stop = fp._padded_time(time, ik_all)
    padded_ik = interpolate_table(ik_all, padded_time)
    com_padded, mass = center_of_mass(model_path, padded_ik)
    com_padded = fp._smooth_com(com_padded, padded_time, DEFAULT.com_smoothing_alpha)
    pad_dt = float(np.median(np.diff(padded_time)))
    acceleration_padded = np.column_stack([
        np.gradient(np.gradient(com_padded[:, axis], pad_dt), pad_dt) for axis in range(3)
    ])
    translational_padded = mass * (acceleration_padded - np.array([0, -DEFAULT.gravity_mps2, 0]))
    rotation_com_padded, momentum_rate_padded = center_of_mass_and_momentum_rate(
        model_path, ik_all, padded_time, DEFAULT.id_lowpass_hz
    )
    return (translational_padded[core_start:core_stop], rotation_com_padded[core_start:core_stop],
            momentum_rate_padded[core_start:core_stop], float(mass))


def _safe_stat(function, values, default=np.nan):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(function(values)) if values.size else default


def _metrics_for_trial(trial, subject, activity, fp, calibration_obj):
    trc = read_trc(trial.effective_opencap_trc())
    ik_all = read_storage(trial.effective_opencap_ik())
    insole_all = read_insole(trial.insole)
    time, insole, _ik, _sync = activity.synchronize(trial, trc, ik_all, insole_all, subject.rules)

    calibration = calibration_obj.load(subject)
    geometry = InsoleGeometry.for_subject(subject)
    translational, rotation_com, momentum_rate, mass = _com_and_rotation(
        trial.opencap_model, ik_all, time, fp
    )

    side = trial.side
    own_column = "right_force" if side == "R" else "left_force"
    own_insole = insole[own_column].to_numpy(float)
    contact = activity.contact(own_insole)

    placement = FrameworkProcessor._placement_for(calibration, trial, side)
    cop = geometry.transform(trc, time, insole, side, placement)
    lever = cop - rotation_com

    lever_x, lever_y = lever[:, 0], lever[:, 1]
    fx_trans, fy_trans = translational[:, 0], translational[:, 1]
    m_z = momentum_rate[:, 2]

    finite_lever = np.isfinite(lever).all(axis=1)
    active = contact & finite_lever & (np.abs(lever_x) > MIN_AP_LEVER_M)

    fy_rotation = np.full(len(time), np.nan)
    fy_rotation[active] = (m_z[active] + lever_y[active] * fx_trans[active]) / lever_x[active]
    blend = np.where(np.isfinite(fy_rotation), 0.5 * fy_trans + 0.5 * fy_rotation, fy_trans)
    blend = np.where(contact, blend, 0.0)
    translational_only = np.where(contact, fy_trans, 0.0)

    # Ground truth, on the framework clock (outcomes.py's own shift convention).
    force_plate = read_storage(trial.force_mot)
    plate_number = 1 if side == "R" else 2
    plate_raw = np.abs(force_plate[f"ground_force_{plate_number}_vy"].to_numpy(float))
    shift = float(time[np.argmax(own_insole)] - force_plate.time.iloc[np.argmax(plate_raw)])
    plate = np.interp(time, force_plate.time + shift, plate_raw, left=np.nan, right=np.nan)

    valid = contact & np.isfinite(plate)
    valid_active = valid & active

    def rmse(estimate, mask):
        if mask.sum() < 5:
            return np.nan
        return float(np.sqrt(np.mean((np.abs(estimate[mask]) - plate[mask]) ** 2)))

    rmse_blend = rmse(blend, valid)
    rmse_translational = rmse(translational_only, valid)
    rmse_blend_active = rmse(blend, valid_active)
    rmse_translational_active = rmse(translational_only, valid_active)

    # Candidate predictors -- every one computable WITHOUT the force plate.
    contact_lever_x = np.abs(lever_x[contact & finite_lever])
    contact_lever_y = np.abs(lever_y[contact & finite_lever])
    amplification = np.divide(
        contact_lever_y, contact_lever_x,
        out=np.full(contact_lever_x.shape, np.inf), where=contact_lever_x > 1e-6,
    )
    fy_share_term = np.abs(lever_x * fy_trans)
    fx_share_term = np.abs(lever_y * fx_trans)
    total_term = fy_share_term + fx_share_term
    identifiability = np.divide(
        fy_share_term, total_term, out=np.full(total_term.shape, np.nan), where=total_term > 1e-9
    )[contact & finite_lever]

    contact_int = contact.astype(int)
    bouts = int((np.diff(contact_int) == 1).sum() + (1 if contact_int[0] else 0))
    active_int = active.astype(int)
    crossings = int(np.abs(np.diff(active_int)).sum())
    dt = float(np.median(np.diff(time)))

    peak_plate = _safe_stat(np.max, plate[valid]) if valid.any() else np.nan
    row = {
        "Subject": subject.id, "Trial": trial.name, "Side": side,
        "ContactFrames": int(contact.sum()),
        "ContactDuration_s": round(float(contact.sum()) * dt, 4),
        "ContactBouts": bouts,
        "Mass_kg": round(mass, 3),
        # --- lever-arm conditioning (the denominator) ---
        "LeverX_abs_median_m": round(_safe_stat(np.median, contact_lever_x), 5),
        "LeverX_abs_p75_m": round(_safe_stat(lambda v: np.percentile(v, 75), contact_lever_x), 5),
        "LeverX_abs_max_m": round(_safe_stat(np.max, contact_lever_x), 5),
        "FracLeverX_gt_05": round(float(np.mean(contact_lever_x > 0.05)) if contact_lever_x.size else np.nan, 4),
        "FracLeverX_gt_10": round(float(np.mean(contact_lever_x > 0.10)) if contact_lever_x.size else np.nan, 4),
        "FracLeverX_gt_15": round(float(np.mean(contact_lever_x > 0.15)) if contact_lever_x.size else np.nan, 4),
        # --- error amplification onto the noisy Fx channel ---
        "Amplification_median": round(_safe_stat(np.median, amplification[np.isfinite(amplification)]), 3),
        "Amplification_p25": round(
            _safe_stat(lambda v: np.percentile(v, 25), amplification[np.isfinite(amplification)]), 3),
        # --- how much of the sagittal moment Fy actually explains ---
        "Identifiability_median": round(_safe_stat(np.median, identifiability), 4),
        "Identifiability_p75": round(_safe_stat(lambda v: np.percentile(v, 75), identifiability), 4),
        # --- how much the H-dot term is even switched on, and how choppily ---
        "ActiveFraction": round(float(active.sum()) / max(int(contact.sum()), 1), 4),
        "ActiveCrossings": crossings,
        "Mz_abs_median_Nm": round(_safe_stat(np.median, np.abs(m_z[contact])), 3),
        "PeakFyTranslational_N": round(_safe_stat(np.max, np.abs(fy_trans[contact])), 1),
        # --- ground-truth outcomes (validation only; never used by the rule) ---
        "PeakPlate_N": round(peak_plate, 1),
        "RMSE_blend_N": round(rmse_blend, 1),
        "RMSE_translational_N": round(rmse_translational, 1),
        "RMSE_blend_active_N": round(rmse_blend_active, 1),
        "RMSE_translational_active_N": round(rmse_translational_active, 1),
        "DeltaRMSE_all_N": round(rmse_translational - rmse_blend, 1),
        "DeltaRMSE_active_N": round(rmse_translational_active - rmse_blend_active, 1),
    }

    cache = RESULTS / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache / f"{trial.name}.npz",
        time=time, lever=lever, translational=translational, momentum_rate=momentum_rate,
        contact=contact, active=active, plate=plate, blend=blend,
        translational_only=translational_only, own_insole=own_insole, mass=mass,
    )
    return row


def collect(subject_ids):
    activity = activity_for(ACTIVITY_NAME, DEFAULT)
    discovery = TrialDiscovery(DEFAULT)
    fp = FrameworkProcessor(DEFAULT)
    calibration_obj = InsoleCalibration(DEFAULT)
    rows = []

    for subject_id in subject_ids:
        subject = Subject(subject_id, DEFAULT)
        if not (DEFAULT.calibration_root / f"{subject_id}_insole_marker_foot_calibration.json").is_file():
            print(f"{subject_id}: no calibration file -- skipping")
            continue
        trials = [
            trial for trial in discovery.trials(subject, activity, "all", for_analysis=True)
            if trial.insole.is_file() and trial.force_mot.is_file()
            and trial.effective_opencap_trc() and Path(trial.effective_opencap_trc()).is_file()
            and trial.effective_opencap_ik() and Path(trial.effective_opencap_ik()).is_file()
        ]
        made = 0
        for trial in sorted(trials, key=lambda t: t.name):
            try:
                rows.append(_metrics_for_trial(trial, subject, activity, fp, calibration_obj))
                made += 1
            except Exception as error:  # noqa: BLE001 -- report and keep going per trial
                print(f"  {trial.name}: FAILED ({type(error).__name__}: {error})")
        print(f"{subject_id}: {made}/{len(trials)} trials")

    if not rows:
        print("No trials collected.")
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "trial_metrics.csv"
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} trials -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", help="Just this one subject instead of all 6")
    args = parser.parse_args()
    collect((args.subject,) if args.subject else SUBJECT_IDS)


if __name__ == "__main__":
    main()
