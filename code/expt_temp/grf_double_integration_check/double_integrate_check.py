"""double_integrate_check.py -- physics-consistency check on the framework
pipeline's reported GRF: double-integrate acceleration derived from GRF
(a = grf/mass + gravity_vector) and compare the resulting reconstructed
COM trajectory against the actual, independently-measured COM trajectory
(center_of_mass() from IK, already stored in each trial's trial_data.npz
as "com_m") -- if the GRF signal is physically consistent, integrating it
back up should reproduce the real COM displacement.

Why this is informative beyond the peak-GRF comparisons already done this
session: for a unilateral SLJ trial with insole_vertical_weight=1.0,
com_vertical_weight=0.0 (the common case), the reported vertical GRF
(grf_N[:,1]) comes PURELY from the calibrated insole reading -- an
INDEPENDENT measurement from the COM trajectory used here as ground
truth (com_m, from marker-free OpenCap kinematics). So this check is a
genuine, independent cross-validation of the insole-derived vertical GRF's
shape/magnitude, not circular -- unlike the horizontal axes (Fx/Fz), which
already come from double-DIFFERENTIATING this same COM trajectory
(framework_processor.py's own total_grf computation), so integrating them
back up is expected to match closely by construction and mostly checks
the differentiate/integrate round-trip's own numerical error, not GRF
correctness.

Initial conditions: p(t0) = com_m[0] (measured), v(t0) estimated by
central-differencing the first few measured COM samples (the best
available estimate of true initial velocity -- there is no other
source). Integration errors accumulate over time (a real property of
double integration, not a testing artifact) -- that's expected and part
of what the RMSE/final-displacement-error columns are meant to show.

Run (from code/, in the opensimPy environment):
    python expt_temp/grf_double_integration_check/double_integrate_check.py
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

SUBJECT_IDS = ("AB03", "AB05", "AB07", "AB08", "AB10", "NTS01")
ACTIVITY_NAME = "single_leg_jump"
AXES = ("Fx (AP)", "Fy (vertical)", "Fz (ML)")
OUT_ROOT = Path(__file__).resolve().parent / "results"


def _rms(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values ** 2))) if len(values) else np.nan


def reconstruct_com(time, grf_N, mass_kg, com_m, gravity_mps2):
    """Double-integrate GRF-derived acceleration; return reconstructed
    position (N, 3), matched to com_m's own initial position/velocity."""
    gravity_vec = np.array([0.0, -gravity_mps2, 0.0])
    acceleration = grf_N / mass_kg + gravity_vec

    # Initial velocity: central difference of the first few real COM
    # samples (best available estimate; not derivable from GRF alone).
    dt0 = time[1] - time[0]
    v0 = (com_m[1] - com_m[0]) / dt0

    velocity = np.zeros_like(acceleration)
    velocity[0] = v0
    velocity[1:] = v0 + np.cumsum((acceleration[1:] + acceleration[:-1]) / 2 * np.diff(time)[:, None], axis=0)

    position = np.zeros_like(velocity)
    position[0] = com_m[0]
    position[1:] = com_m[0] + np.cumsum((velocity[1:] + velocity[:-1]) / 2 * np.diff(time)[:, None], axis=0)
    return position


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

    reconstructed = reconstruct_com(time, grf_N, mass_kg, com_m, gravity_mps2)
    error = reconstructed - com_m

    row = {"Trial": trial.name, "Condition": trial.condition, "Side": trial.side}
    for axis_index, axis_label in enumerate(AXES):
        row[f"{axis_label}_RMSE_mm"] = round(1000 * _rms(error[:, axis_index]), 2)
        row[f"{axis_label}_ActualDisp_mm"] = round(1000 * (com_m[-1, axis_index] - com_m[0, axis_index]), 2)
        row[f"{axis_label}_ReconDisp_mm"] = round(1000 * (reconstructed[-1, axis_index] - com_m[0, axis_index]), 2)
    return row, time, com_m, reconstructed


def plot_trial(subject_id, trial_name, time, com_m, reconstructed, out_dir):
    x = time - time[0]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis_index, axis_label in enumerate(AXES):
        ax = axes[axis_index]
        ax.plot(x, 1000 * (com_m[:, axis_index] - com_m[0, axis_index]), "k", label="Actual (kinematics)")
        ax.plot(x, 1000 * (reconstructed[:, axis_index] - com_m[0, axis_index]), "tab:red",
                label="Reconstructed (double-integrated GRF)")
        ax.set_title(axis_label)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("COM displacement (mm)")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    figure.suptitle(f"{trial_name} -- GRF double-integration vs. actual COM displacement")
    figure.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_dir / f"{trial_name}_double_integration.png", dpi=150)
    plt.close(figure)


def main():
    discovery = TrialDiscovery(DEFAULT)
    activity = activity_for(ACTIVITY_NAME, DEFAULT)

    for subject_id in SUBJECT_IDS:
        subject = Subject(subject_id, DEFAULT)
        trials = discovery.trials(subject, activity, "all", for_analysis=True)
        rows = []
        subject_out = OUT_ROOT / subject_id
        for trial in trials:
            result = analyze_trial(trial, DEFAULT.gravity_mps2)
            if result is None:
                continue
            row, time, com_m, reconstructed = result
            rows.append(row)
            plot_trial(subject_id, trial.name, time, com_m, reconstructed, subject_out)

        if not rows:
            print(f"{subject_id}: no valid trials -- skipped")
            continue
        table = pd.DataFrame(rows)
        subject_out.mkdir(parents=True, exist_ok=True)
        out_path = subject_out / f"{subject_id}_double_integration_summary.csv"
        table.to_csv(out_path, index=False)
        print(f"{subject_id}: {len(table)} trial(s) -> {out_path}")
        vertical_rmse = table["Fy (vertical)_RMSE_mm"]
        print(f"  Fy (vertical, insole-only) RMSE: mean={vertical_rmse.mean():.1f}mm, "
              f"max={vertical_rmse.max():.1f}mm")


if __name__ == "__main__":
    main()
