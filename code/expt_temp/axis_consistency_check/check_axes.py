"""check_axes.py -- check whether marker-based and OpenCap-based pipelines
agree on which axis is AP (X, forward) vs ML (Z, lateral), by comparing
(1) COM trajectories and (2) the raw pelvis translation coordinates
(pelvis_tx/pelvis_tz) from each pipeline's own IK solution, aligned onto a
common clock (same peak-matching heuristic used throughout this project).

If X and Z are consistent between pipelines, X-vs-X and Z-vs-Z should
correlate much more strongly than the cross pairs (X-vs-Z, Z-vs-X). If
they're swapped for some subject/trial, the cross pairs would correlate
better instead.

Run (from code/, in the opensimPy environment):
    python expt_temp/axis_consistency_check/check_axes.py <trial_name>
"""
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
from discovery import TrialDiscovery  # noqa: E402
from activities import activity_for  # noqa: E402
from subject import Subject  # noqa: E402
from io_tables import read_storage  # noqa: E402
from opensim_env import center_of_mass  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "results"


def _interp(source_time, values, target_time):
    return np.interp(target_time, source_time, values, left=np.nan, right=np.nan)


def analyze(trial_name: str):
    subject_id = trial_name.split("_")[0]
    subject = Subject(subject_id, DEFAULT)
    activity = activity_for("single_leg_jump", DEFAULT)
    discovery = TrialDiscovery(DEFAULT)
    trial = next(t for t in discovery.trials(subject, activity, "all", for_analysis=True) if t.name == trial_name)

    marker_root = DEFAULT.results_root / subject_id / "marker_based" / trial.condition
    marker_ik_path = marker_root / "ik" / f"{trial_name}_ik.mot"
    marker_model_path = (
        DEFAULT.results_root / subject_id / "marker_based" / "models"
        / f"{subject_id}_shoe_size_{subject.shoe_size}_{trial.condition}_scaled.osim"
    )
    marker_ik = read_storage(marker_ik_path)
    marker_com, _ = center_of_mass(marker_model_path, marker_ik)

    opencap_ik_path = trial.effective_opencap_ik()
    opencap_ik = read_storage(opencap_ik_path)
    opencap_com, _ = center_of_mass(trial.opencap_model, opencap_ik)

    # Align marker clock onto the framework/OpenCap clock via the SAME
    # peak-matching heuristic outcomes.py already uses elsewhere in this
    # project (framework force peak vs. marker force-plate peak).
    package = trial.output / "trial_data.npz"
    data = np.load(package, allow_pickle=True)
    framework_time = data["time"].astype(float)
    framework_force = data["force"].astype(float)
    force_mot = DEFAULT.data_root / subject_id / "PROCESSED" / "MOCAP" / "force_data" / f"{trial_name}_forces_PROCESSED.mot"
    force_plate = read_storage(force_mot)
    plate = 1 if trial.side == "R" else 2
    marker_vertical = np.abs(force_plate[f"ground_force_{plate}_vy"].to_numpy(float))
    shift = float(framework_time[np.argmax(framework_force)] - force_plate.time.iloc[np.argmax(marker_vertical)])

    common_time = opencap_ik["time"].to_numpy(float)
    common_time = common_time[(common_time >= opencap_ik.time.min()) & (common_time <= opencap_ik.time.max())]

    marker_time_shifted = marker_ik["time"].to_numpy(float) + shift
    marker_com_on_common = np.column_stack([
        _interp(marker_time_shifted, marker_com[:, axis], common_time) for axis in range(3)
    ])
    opencap_com_on_common = np.column_stack([
        _interp(opencap_ik["time"].to_numpy(float), opencap_com[:, axis], common_time) for axis in range(3)
    ])

    marker_tx = _interp(marker_time_shifted, marker_ik["pelvis_tx"].to_numpy(float), common_time)
    marker_tz = _interp(marker_time_shifted, marker_ik["pelvis_tz"].to_numpy(float), common_time)
    opencap_tx = _interp(opencap_ik["time"].to_numpy(float), opencap_ik["pelvis_tx"].to_numpy(float), common_time)
    opencap_tz = _interp(opencap_ik["time"].to_numpy(float), opencap_ik["pelvis_tz"].to_numpy(float), common_time)

    def corr(a, b):
        mask = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[mask], b[mask])[0, 1]) if mask.sum() > 2 else np.nan

    print(f"=== {trial_name} (side={trial.side}, shift={shift:.3f}s) ===")
    print("COM correlations (marker axis vs opencap axis):")
    labels = ["X (AP)", "Y (vert)", "Z (ML)"]
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            c = corr(marker_com_on_common[:, i], opencap_com_on_common[:, j])
            print(f"  marker_{li:8s} vs opencap_{lj:8s}: r={c:+.3f}")

    print("\nPelvis translation correlations (marker vs opencap):")
    print(f"  marker_tx vs opencap_tx: r={corr(marker_tx, opencap_tx):+.3f}")
    print(f"  marker_tx vs opencap_tz: r={corr(marker_tx, opencap_tz):+.3f}")
    print(f"  marker_tz vs opencap_tx: r={corr(marker_tz, opencap_tx):+.3f}")
    print(f"  marker_tz vs opencap_tz: r={corr(marker_tz, opencap_tz):+.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x = common_time - common_time[0]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(x, marker_com_on_common[:, 0] - np.nanmean(marker_com_on_common[:, 0]), "k", label="Marker X (AP)")
    axes[0, 0].plot(x, opencap_com_on_common[:, 0] - np.nanmean(opencap_com_on_common[:, 0]), "tab:red", label="OpenCap X (AP)")
    axes[0, 0].set_title("COM X (AP), demeaned")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(x, marker_com_on_common[:, 2] - np.nanmean(marker_com_on_common[:, 2]), "k", label="Marker Z (ML)")
    axes[0, 1].plot(x, opencap_com_on_common[:, 2] - np.nanmean(opencap_com_on_common[:, 2]), "tab:red", label="OpenCap Z (ML)")
    axes[0, 1].set_title("COM Z (ML), demeaned")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(x, marker_tx - np.nanmean(marker_tx), "k", label="Marker pelvis_tx")
    axes[1, 0].plot(x, opencap_tx - np.nanmean(opencap_tx), "tab:red", label="OpenCap pelvis_tx")
    axes[1, 0].set_title("pelvis_tx, demeaned")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(x, marker_tz - np.nanmean(marker_tz), "k", label="Marker pelvis_tz")
    axes[1, 1].plot(x, opencap_tz - np.nanmean(opencap_tz), "tab:red", label="OpenCap pelvis_tz")
    axes[1, 1].set_title("pelvis_tz, demeaned")
    axes[1, 1].legend(fontsize=8)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("meters")
            ax.grid(True, alpha=0.3)
    figure.suptitle(f"{trial_name}: marker vs. OpenCap axis consistency check")
    figure.tight_layout()
    figure.savefig(OUT_DIR / f"{trial_name}_axis_check.png", dpi=150)
    plt.close(figure)
    print(f"\nSaved -> {OUT_DIR / f'{trial_name}_axis_check.png'}")


if __name__ == "__main__":
    analyze(sys.argv[1])
