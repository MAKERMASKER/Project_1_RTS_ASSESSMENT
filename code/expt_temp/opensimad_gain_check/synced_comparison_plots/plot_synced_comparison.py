"""plot_synced_comparison.py -- for the solved push-off (0.6s) window, plot
all three signals on one common, properly-synced time axis:
  - Ground truth (marker force plate vertical GRF)
  - Framework/production (raw insole vertical force x current production gain)
  - OpenSimAD (kinematics-only physics estimate)

All three are resampled onto OpenSimAD's own sample times (already on the
framework/OpenCap clock), using the SAME synchronization each pipeline
already established (marker<->framework peak-matching shift; insole<->
OpenCap shift_s from activity.synchronize()) -- no new alignment method,
just a clear, single plot per trial showing what was previously only
available as separate numbers.

Run (from code/, in the opensimPy environment):
    python expt_temp/opensimad_gain_check/synced_comparison_plots/plot_synced_comparison.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_ROOT))

from config import DEFAULT  # noqa: E402
from io_tables import read_storage, read_insole  # noqa: E402
from subject import Subject  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from activities import activity_for  # noqa: E402

PLAN_PATH = CODE_ROOT / "expt_temp" / "opensimad_gain_check" / "trial_plan_pushoff.json"
RESULTS_DIR = CODE_ROOT / "expt_temp" / "opensimad_gain_check" / "results"
OUT_DIR = Path(__file__).resolve().parent / "results"

RESULT_OVERRIDES = {
    ("AB03", "L"): "AB03_NOEXO_SLJ_L_3_pushoff_v2",
}


def plot_trial(entry, discovery, activity):
    subject_id, side, trial_name = entry["subject"], entry["side"], entry["trial_name"]
    window = entry["time_window"]
    result_name = RESULT_OVERRIDES.get((subject_id, side), f"{trial_name}_pushoff")

    grf_path = RESULTS_DIR / result_name / "GRF.csv"
    status = "converged"
    if not grf_path.is_file():
        grf_path = RESULTS_DIR / result_name / "GRF_NonConvergent.csv"
        status = "non-convergent"
    if not grf_path.is_file():
        print(f"{trial_name}: no OpenSimAD result yet -- skipped")
        return

    grf = pd.read_csv(grf_path)
    word = "right" if side == "R" else "left"
    opensimad_time = grf["time"].to_numpy(float)
    opensimad_vertical = np.abs(grf[f"ground_force_{word}_vy"].to_numpy(float))
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
    shift_marker = float(
        framework_time[np.argmax(framework_force)] - force_plate.time.iloc[np.argmax(marker_vertical_full)]
    )
    marker_time_shifted = force_plate.time.to_numpy(float) + shift_marker
    marker_on_common = np.interp(common_time, marker_time_shifted, marker_vertical_full, left=np.nan, right=np.nan)

    insole_all = read_insole(trial.insole)
    insole_time_shifted = insole_all.time.to_numpy(float) + shift_insole
    column = "right_force" if side == "R" else "left_force"
    raw_insole = insole_all[column].to_numpy(float)
    raw_on_common = np.interp(common_time, insole_time_shifted, raw_insole, left=np.nan, right=np.nan)

    calibration = json.loads(
        (DEFAULT.calibration_root / f"{subject_id}_insole_marker_foot_calibration.json").read_text()
    )
    production_gain = calibration[side]["vertical_force_scale"]
    framework_on_common = raw_on_common * production_gain

    x = common_time - window[0]
    figure, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, marker_on_common, color="black", linewidth=2.2, label="Ground truth (marker force plate)")
    ax.plot(x, framework_on_common, color="tab:blue", linewidth=1.8, label=f"Framework/production (gain={production_gain:.3f})")
    ax.plot(x, opensimad_on_common, color="tab:red", linewidth=1.8, label=f"OpenSimAD ({status})")
    ax.set_xlabel("Time since window start (s)")
    ax.set_ylabel("Vertical GRF (N)")
    ax.set_title(f"{trial_name} (side={side}) -- push-off window [{window[0]:.2f}, {window[1]:.2f}]s, synced")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    figure.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{trial_name}_synced_pushoff.png"
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"{trial_name}: saved -> {out_path}")


def main():
    plan = json.loads(PLAN_PATH.read_text())
    discovery = TrialDiscovery(DEFAULT)
    activity = activity_for("single_leg_jump", DEFAULT)
    for entry in plan:
        plot_trial(entry, discovery, activity)


if __name__ == "__main__":
    main()
