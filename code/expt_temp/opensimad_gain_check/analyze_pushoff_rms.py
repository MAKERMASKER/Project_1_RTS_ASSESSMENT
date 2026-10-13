"""analyze_pushoff_rms.py -- corrected version of analyze_pushoff_batch.py.

The earlier peak-only comparison (OpenSimAD's single max value vs.
marker's single max value) is misleading: a single-sample peak can agree
by coincidence (or disagree due to a few-millisecond timing offset) while
the rest of the trace disagrees badly, or vice versa. This version
compares the FULL traces over the whole push-off window instead:

- OpenSimAD accuracy: RMSE between OpenSimAD's vertical GRF trace and the
  marker force-plate's vertical GRF trace, both resampled onto a common
  time grid spanning the window (not a single peak-to-peak difference).
- Gain: a least-squares (regression-through-origin) fit of the raw insole
  trace against the reference trace over the whole window --
  `gain = sum(raw * reference) / sum(raw**2)` -- rather than a single
  peak ratio, so one bad instant can't dominate the estimate the way a
  peak-ratio does.

Run (from code/, in the opensimPy environment):
    python expt_temp/opensimad_gain_check/analyze_pushoff_rms.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from config import DEFAULT  # noqa: E402
from io_tables import read_storage, read_insole  # noqa: E402
from subject import Subject  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from activities import activity_for  # noqa: E402

PLAN_PATH = Path(__file__).resolve().parent / "trial_plan_pushoff.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# (result_folder_suffix, trial_name override) -- lets a rerun/rework result
# override the original one for a given (subject, side) without editing
# trial_plan_pushoff.json. None means "use the plan's own trial_name".
RESULT_OVERRIDES = {
    ("AB03", "L"): "AB03_NOEXO_SLJ_L_3_pushoff_v2",
}


def _rms(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values ** 2))) if len(values) else np.nan


def marker_trace(subject_id, trial_name, side, window, common_time, shift):
    force_mot = DEFAULT.data_root / subject_id / "PROCESSED" / "MOCAP" / "force_data" / f"{trial_name}_forces_PROCESSED.mot"
    force_plate = read_storage(force_mot)
    plate = 1 if side == "R" else 2
    marker_vertical = np.abs(force_plate[f"ground_force_{plate}_vy"].to_numpy(float))
    marker_time_shifted = force_plate.time.to_numpy(float) + shift
    return np.interp(common_time, marker_time_shifted, marker_vertical, left=np.nan, right=np.nan)


def raw_insole_trace(subject, trial, side, common_time, shift_insole):
    insole_all = read_insole(trial.insole)
    insole_time = insole_all.time.to_numpy(float) + shift_insole
    column = "right_force" if side == "R" else "left_force"
    own_force = insole_all[column].to_numpy(float)
    return np.interp(common_time, insole_time, own_force, left=np.nan, right=np.nan)


def analyze(entry, discovery, activity):
    subject_id, side, trial_name = entry["subject"], entry["side"], entry["trial_name"]
    window = entry["time_window"]
    result_name = RESULT_OVERRIDES.get((subject_id, side), f"{trial_name}_pushoff")

    grf_path = RESULTS_DIR / result_name / "GRF.csv"
    status = "converged"
    if not grf_path.is_file():
        grf_path = RESULTS_DIR / result_name / "GRF_NonConvergent.csv"
        status = "non_convergent"
    if not grf_path.is_file():
        return None

    grf = pd.read_csv(grf_path)
    word = "right" if side == "R" else "left"
    opensimad_time = grf["time"].to_numpy(float)
    opensimad_vertical = np.abs(grf[f"ground_force_{word}_vy"].to_numpy(float))

    # Common time grid: OpenSimAD's own sample times (already on the
    # framework/OpenCap clock -- same time_window we requested), clipped
    # to the actual window.
    common_time = opensimad_time[(opensimad_time >= window[0]) & (opensimad_time <= window[1])]
    opensimad_on_common = np.interp(common_time, opensimad_time, opensimad_vertical)

    subject = Subject(subject_id, DEFAULT)
    trial = next(t for t in discovery.trials(subject, activity, "NOEXO", for_analysis=True) if t.name == trial_name)
    package = trial.output / "trial_data.npz"
    data = np.load(package, allow_pickle=True)
    framework_time = data["time"].astype(float)
    framework_force = data["force"].astype(float)
    shift_insole = float(data["shift_s"])

    force_mot = DEFAULT.data_root / subject_id / "PROCESSED" / "MOCAP" / "force_data" / f"{trial_name}_forces_PROCESSED.mot"
    force_plate = read_storage(force_mot)
    plate = 1 if side == "R" else 2
    marker_vertical_full = np.abs(force_plate[f"ground_force_{plate}_vy"].to_numpy(float))
    shift_marker = float(framework_time[np.argmax(framework_force)] - force_plate.time.iloc[np.argmax(marker_vertical_full)])

    marker_on_common = marker_trace(subject_id, trial_name, side, window, common_time, shift_marker)
    raw_on_common = raw_insole_trace(subject, trial, side, common_time, shift_insole)

    mask = np.isfinite(opensimad_on_common) & np.isfinite(marker_on_common) & np.isfinite(raw_on_common)
    if mask.sum() < 3:
        return None

    opensimad_rmse = _rms(opensimad_on_common[mask] - marker_on_common[mask])
    opensimad_rmse_pct = 100 * opensimad_rmse / np.mean(marker_on_common[mask])

    # Least-squares (regression-through-origin) gain: minimizes
    # sum((gain*raw - reference)**2) over the WHOLE window, not a single
    # peak ratio -- reference is marker ground truth (the "ideal" gain,
    # analogous to before) and OpenSimAD (the ground-truth-free estimate).
    raw = raw_on_common[mask]
    marker_ref = marker_on_common[mask]
    opensimad_ref = opensimad_on_common[mask]
    marker_gain = float(np.sum(raw * marker_ref) / np.sum(raw ** 2))
    opensimad_gain = float(np.sum(raw * opensimad_ref) / np.sum(raw ** 2))

    # RMSE of the gain-corrected insole trace vs. marker, using EACH gain
    # -- shows how well the whole trace (not just the peak) matches once
    # scaled, for both the "true" and OpenSimAD-derived gain.
    corrected_with_marker_gain_rmse = _rms(marker_gain * raw - marker_ref)
    corrected_with_opensimad_gain_rmse = _rms(opensimad_gain * raw - marker_ref)

    return {
        "Trial": trial_name, "Side": side, "Status": status, "N_samples": int(mask.sum()),
        "OpenSimAD_vs_Marker_RMSE_N": round(opensimad_rmse, 1),
        "OpenSimAD_vs_Marker_RMSE_pct": round(opensimad_rmse_pct, 1),
        "MarkerLSQGain": round(marker_gain, 3),
        "OpenSimADLSQGain": round(opensimad_gain, 3),
        "GainPctDiff": round(100 * (opensimad_gain - marker_gain) / marker_gain, 1),
        "TraceRMSE_with_MarkerGain_N": round(corrected_with_marker_gain_rmse, 1),
        "TraceRMSE_with_OpenSimADGain_N": round(corrected_with_opensimad_gain_rmse, 1),
    }


def main():
    plan = json.loads(PLAN_PATH.read_text())
    discovery = TrialDiscovery(DEFAULT)
    activity = activity_for("single_leg_jump", DEFAULT)

    rows = []
    for entry in plan:
        result = analyze(entry, discovery, activity)
        if result is None:
            print(f"{entry['trial_name']}: no result yet -- skipped")
            continue
        rows.append(result)

    table = pd.DataFrame(rows)
    out_path = Path(__file__).resolve().parent / "pushoff_rms_comparison.csv"
    table.to_csv(out_path, index=False)
    print(table.to_string(index=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
