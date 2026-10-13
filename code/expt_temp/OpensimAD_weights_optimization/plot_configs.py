"""plot_configs.py -- overlay ground truth, framework/production, and all
4 weight configs' OpenSimAD traces on one synced plot for the pilot trial,
so the "does the early dip-then-rise wobble go away" question can be
answered visually. Extends
../opensimad_gain_check/synced_comparison_plots/plot_synced_comparison.py's
approach (same synchronization, more traces).

Run (from code/, in the opensimPy environment):
    python expt_temp/OpensimAD_weights_optimization/plot_configs.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

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

CONFIG_COLORS = {
    "config0_baseline": "tab:red",
    "config1_accel500": "tab:orange",
    "config2_activation01": "tab:green",
    "config3_combined": "tab:purple",
}
CONFIG_LABELS = {
    "config0_baseline": "OpenSimAD baseline (accel=50, activation=1)",
    "config1_accel500": "OpenSimAD accel=500",
    "config2_activation01": "OpenSimAD activation=0.1",
    "config3_combined": "OpenSimAD accel=500, activation=0.1",
}


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
    marker_time_shifted = force_plate.time.to_numpy(float) + shift_marker

    insole_all = read_insole(trial.insole)
    insole_time_shifted = insole_all.time.to_numpy(float) + shift_insole
    column = "right_force" if SIDE == "R" else "left_force"
    raw_insole = insole_all[column].to_numpy(float)

    calibration = json.loads(
        (DEFAULT.calibration_root / f"{SUBJECT_ID}_insole_marker_foot_calibration.json").read_text()
    )
    production_gain = calibration[SIDE]["vertical_force_scale"]

    figure, ax = plt.subplots(figsize=(11, 7))

    # Common time grid for reference traces: OpenSimAD's own sample times
    # from any available config (all share the same window).
    any_grf = None
    for config_name in CONFIG_COLORS:
        grf_path = RESULTS_DIR / config_name / "GRF.csv"
        if grf_path.is_file():
            any_grf = pd.read_csv(grf_path)
            break
    if any_grf is None:
        raise SystemExit("No config results available yet")
    opensimad_time_ref = any_grf["time"].to_numpy(float)
    common_time = opensimad_time_ref[(opensimad_time_ref >= WINDOW[0]) & (opensimad_time_ref <= WINDOW[1])]
    x = common_time - WINDOW[0]

    marker_on_common = np.interp(common_time, marker_time_shifted, marker_vertical_full, left=np.nan, right=np.nan)
    raw_on_common = np.interp(common_time, insole_time_shifted, raw_insole, left=np.nan, right=np.nan)
    framework_on_common = raw_on_common * production_gain

    ax.plot(x, marker_on_common, color="black", linewidth=2.5, label="Ground truth (marker)", zorder=10)
    ax.plot(x, framework_on_common, color="tab:blue", linewidth=2.0, label=f"Framework/production (gain={production_gain:.3f})")

    for config_name, color in CONFIG_COLORS.items():
        grf_path = RESULTS_DIR / config_name / "GRF.csv"
        status = "converged"
        if not grf_path.is_file():
            grf_path = RESULTS_DIR / config_name / "GRF_NonConvergent.csv"
            status = "non-convergent"
        if not grf_path.is_file():
            print(f"{config_name}: no result yet -- skipped in plot")
            continue
        grf = pd.read_csv(grf_path)
        word = "right" if SIDE == "R" else "left"
        opensimad_time = grf["time"].to_numpy(float)
        opensimad_vertical = np.abs(grf[f"ground_force_{word}_vy"].to_numpy(float))
        opensimad_on_common = np.interp(common_time, opensimad_time, opensimad_vertical)
        label = CONFIG_LABELS[config_name] + (" [non-convergent]" if status == "non-convergent" else "")
        ax.plot(x, opensimad_on_common, color=color, linewidth=1.6, linestyle="--", label=label)

    ax.set_xlabel("Time since window start (s)")
    ax.set_ylabel("Vertical GRF (N)")
    ax.set_title(f"{TRIAL_NAME} (side={SIDE}) -- weight-config comparison, push-off window [{WINDOW[0]:.2f}, {WINDOW[1]:.2f}]s")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    figure.tight_layout()

    out_path = RESULTS_DIR / f"{TRIAL_NAME}_all_configs_comparison.png"
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
