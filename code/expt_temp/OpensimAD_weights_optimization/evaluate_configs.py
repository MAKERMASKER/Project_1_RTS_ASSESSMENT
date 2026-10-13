"""evaluate_configs.py -- outer-objective evaluation: whole-trace RMSE of
each weight config's OpenSimAD GRF vs. real marker ground truth, for the
one pilot trial (AB03_NOEXO_SLJ_R_1). Reuses the exact same common-time-
grid + marker-shift methodology as
../opensimad_gain_check/analyze_pushoff_rms.py (not re-derived).

Run (from code/, in the opensimPy environment):
    python expt_temp/OpensimAD_weights_optimization/evaluate_configs.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "expt_temp" / "opensimad_gain_check"))

from config import DEFAULT  # noqa: E402
from io_tables import read_storage, read_insole  # noqa: E402
from subject import Subject  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from activities import activity_for  # noqa: E402

TRIAL_NAME = "AB03_NOEXO_SLJ_R_1"
SIDE = "R"
WINDOW = [3.24, 3.84]
SUBJECT_ID = "AB03"

RESULTS_DIR = Path(__file__).resolve().parent / "results"

CONFIGS = {
    "config0_baseline": {"positionTrackingTerm": 100, "velocityTrackingTerm": 10,
                          "accelerationTrackingTerm": 50, "activationTerm": 1},
    "config1_accel500": {"positionTrackingTerm": 100, "velocityTrackingTerm": 10,
                          "accelerationTrackingTerm": 500, "activationTerm": 1},
    "config2_activation01": {"positionTrackingTerm": 100, "velocityTrackingTerm": 10,
                              "accelerationTrackingTerm": 50, "activationTerm": 0.1},
    "config3_combined": {"positionTrackingTerm": 100, "velocityTrackingTerm": 10,
                          "accelerationTrackingTerm": 500, "activationTerm": 0.1},
}


def _rms(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values ** 2))) if len(values) else np.nan


def main():
    discovery = TrialDiscovery(DEFAULT)
    activity = activity_for("single_leg_jump", DEFAULT)
    subject = Subject(SUBJECT_ID, DEFAULT)
    trial = next(t for t in discovery.trials(subject, activity, "NOEXO", for_analysis=True) if t.name == TRIAL_NAME)

    package = trial.output / "trial_data.npz"
    data = np.load(package, allow_pickle=True)
    framework_time = data["time"].astype(float)
    framework_force = data["force"].astype(float)
    shift_insole = float(data["shift_s"])

    force_mot = DEFAULT.data_root / SUBJECT_ID / "PROCESSED" / "MOCAP" / "force_data" / f"{TRIAL_NAME}_forces_PROCESSED.mot"
    force_plate = read_storage(force_mot)
    plate = 1 if SIDE == "R" else 2
    marker_vertical_full = np.abs(force_plate[f"ground_force_{plate}_vy"].to_numpy(float))
    shift_marker = float(
        framework_time[np.argmax(framework_force)] - force_plate.time.iloc[np.argmax(marker_vertical_full)]
    )

    insole_all = read_insole(trial.insole)
    insole_time_shifted = insole_all.time.to_numpy(float) + shift_insole
    column = "right_force" if SIDE == "R" else "left_force"
    raw_insole = insole_all[column].to_numpy(float)

    rows = []
    for config_name, weights in CONFIGS.items():
        config_dir = RESULTS_DIR / config_name
        grf_path = config_dir / "GRF.csv"
        status = "converged"
        if not grf_path.is_file():
            grf_path = config_dir / "GRF_NonConvergent.csv"
            status = "non_convergent"
        if not grf_path.is_file():
            print(f"{config_name}: no result yet -- skipped")
            continue

        grf = pd.read_csv(grf_path)
        word = "right" if SIDE == "R" else "left"
        opensimad_time = grf["time"].to_numpy(float)
        opensimad_vertical = np.abs(grf[f"ground_force_{word}_vy"].to_numpy(float))
        common_time = opensimad_time[(opensimad_time >= WINDOW[0]) & (opensimad_time <= WINDOW[1])]
        opensimad_on_common = np.interp(common_time, opensimad_time, opensimad_vertical)

        marker_time_shifted = force_plate.time.to_numpy(float) + shift_marker
        marker_on_common = np.interp(common_time, marker_time_shifted, marker_vertical_full, left=np.nan, right=np.nan)
        raw_on_common = np.interp(common_time, insole_time_shifted, raw_insole, left=np.nan, right=np.nan)

        mask = np.isfinite(opensimad_on_common) & np.isfinite(marker_on_common) & np.isfinite(raw_on_common)
        opensimad_rmse = _rms(opensimad_on_common[mask] - marker_on_common[mask])
        opensimad_rmse_pct = 100 * opensimad_rmse / np.mean(marker_on_common[mask])

        raw = raw_on_common[mask]
        marker_ref = marker_on_common[mask]
        opensimad_ref = opensimad_on_common[mask]
        marker_gain = float(np.sum(raw * marker_ref) / np.sum(raw ** 2))
        opensimad_gain = float(np.sum(raw * opensimad_ref) / np.sum(raw ** 2))

        rows.append({
            "Config": config_name, "Status": status, **weights,
            "OpenSimAD_vs_Marker_RMSE_N": round(opensimad_rmse, 1),
            "OpenSimAD_vs_Marker_RMSE_pct": round(opensimad_rmse_pct, 1),
            "MarkerLSQGain": round(marker_gain, 3),
            "OpenSimADLSQGain": round(opensimad_gain, 3),
            "GainPctDiff": round(100 * (opensimad_gain - marker_gain) / marker_gain, 1),
        })
        print(f"{config_name}: RMSE={opensimad_rmse_pct:.1f}%  gain_gap={100*(opensimad_gain-marker_gain)/marker_gain:+.1f}%")

    table = pd.DataFrame(rows).sort_values("OpenSimAD_vs_Marker_RMSE_pct")
    out_path = RESULTS_DIR / "config_ranking.csv"
    table.to_csv(out_path, index=False)
    print(f"\n{table.to_string(index=False)}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
