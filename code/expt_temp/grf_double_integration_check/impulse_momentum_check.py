"""impulse_momentum_check.py -- companion to double_integrate_check.py.

Checks the impulse-momentum theorem directly: the net area under the
GRF-derived acceleration curve (a single integration) should equal the
ACTUAL change in COM velocity over the window (impulse / mass = delta-v).
This is a genuinely different, less sensitive check than double
integration -- one integration step accumulates error linearly, not
quadratically, so a trial that looks bad on the double-integration
trajectory RMSE can still look fine here, and vice versa; agreement here
is necessary but not sufficient for displacement to also match.

Also extracts the NET (start-to-end) displacement comparison already
implicit in double_integrate_check.py's per-trial columns (ActualDisp_mm/
ReconDisp_mm) into its own summary -- the "does the net area match the
overall displacement" question as literally asked, distinct from the
full-trajectory RMSE already reported.

Run (from code/, in the opensimPy environment, after
double_integrate_check.py has produced trial_data.npz-backed summaries):
    python expt_temp/grf_double_integration_check/impulse_momentum_check.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from config import DEFAULT  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from activities import activity_for  # noqa: E402
from subject import Subject  # noqa: E402

SUBJECT_IDS = ("AB03", "AB05", "AB07", "AB08", "AB10", "NTS01")
ACTIVITY_NAME = "single_leg_jump"
AXES = ("Fx (AP)", "Fy (vertical)", "Fz (ML)")
OUT_ROOT = Path(__file__).resolve().parent / "results"


def actual_velocity_change(time, com_m):
    """Central-difference velocity estimate at both ends of the window --
    the best available estimate of true delta-v from kinematics alone."""
    dt_start = time[1] - time[0]
    dt_end = time[-1] - time[-2]
    v_start = (com_m[1] - com_m[0]) / dt_start
    v_end = (com_m[-1] - com_m[-2]) / dt_end
    return v_end - v_start


def analyze_trial(trial, gravity_mps2):
    package = trial.output / "trial_data.npz"
    if not package.is_file():
        return None
    data = np.load(package, allow_pickle=True)
    if bool(data.get("quality_omit", False)):
        return None

    time = data["time"].astype(float)
    grf_N = data["grf_N"].astype(float)
    com_m = data["com_m"].astype(float)
    mass_kg = float(data["mass_kg"])

    gravity_vec = np.array([0.0, -gravity_mps2, 0.0])
    acceleration = grf_N / mass_kg + gravity_vec
    # Net area under the acceleration curve (trapezoidal) = impulse / mass = delta-v.
    predicted_dv = np.trapezoid(acceleration, time, axis=0)
    actual_dv = actual_velocity_change(time, com_m)

    row = {"Trial": trial.name, "Condition": trial.condition, "Side": trial.side}
    for axis_index, axis_label in enumerate(AXES):
        row[f"{axis_label}_ActualDeltaV_mps"] = round(float(actual_dv[axis_index]), 3)
        row[f"{axis_label}_PredictedDeltaV_mps"] = round(float(predicted_dv[axis_index]), 3)
        row[f"{axis_label}_DeltaV_Error_mps"] = round(
            float(predicted_dv[axis_index] - actual_dv[axis_index]), 3
        )
    return row


def main():
    discovery = TrialDiscovery(DEFAULT)
    activity = activity_for(ACTIVITY_NAME, DEFAULT)

    all_rows = []
    for subject_id in SUBJECT_IDS:
        subject = Subject(subject_id, DEFAULT)
        trials = discovery.trials(subject, activity, "all", for_analysis=True)
        rows = []
        for trial in trials:
            row = analyze_trial(trial, DEFAULT.gravity_mps2)
            if row is not None:
                rows.append(row)
        if not rows:
            print(f"{subject_id}: no valid trials -- skipped")
            continue
        table = pd.DataFrame(rows)
        subject_out = OUT_ROOT / subject_id
        subject_out.mkdir(parents=True, exist_ok=True)
        out_path = subject_out / f"{subject_id}_impulse_momentum_summary.csv"
        table.to_csv(out_path, index=False)
        vertical_error = table["Fy (vertical)_DeltaV_Error_mps"]
        print(f"{subject_id}: {len(table)} trial(s) -> {out_path}")
        print(f"  Fy (vertical) delta-v error: mean={vertical_error.abs().mean():.3f} m/s, "
              f"max={vertical_error.abs().max():.3f} m/s")
        all_rows.extend(rows)

    combined = pd.DataFrame(all_rows)
    combined.to_csv(OUT_ROOT / "all_subjects_impulse_momentum_summary.csv", index=False)


if __name__ == "__main__":
    main()
