"""collect_insole_debug.py -- debug artifacts for the insole vertical-force
calibration investigation (see subject_rules per-side gain discussion):
for each subject, 2 SLJ trials per side (4 total), raw insole force plotted
against the marker force-plate ground truth, synced onto a common timeline,
over each trial's COMPLETE recorded duration (not cropped to the SLJ
+/-0.75s analysis window) -- so the full stance/flight/landing context is
visible, not just the analysis slice.

No calibration/gain is applied anywhere here -- these are RAW insole
readings (exactly what InsoleCalibration._force_gain() reads before any
scaling), deliberately, since the point is to inspect the raw sensor
behavior that's driving the per-subject calibration inconsistency found
during this investigation.

Output: validation/sensor_insole/shoe_size_<N>/<SUBJECT>/
    <SUBJECT>_raw_insole_vs_marker.png   one image, 4 panels (2 L-side
        trials + 2 R-side trials), raw insole (black) vs marker force-plate
        (tab:red), full trial duration, time-synced.
    <TRIAL_NAME>_raw_vs_marker.csv       per-trial synced time series
        backing each panel (time_s, insole_raw_N, marker_force_N).

Run from code/, in the opensimPy environment:
    python expt_temp/collect_insole_debug.py
    python expt_temp/collect_insole_debug.py --subject AB08
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from activities import activity_for  # noqa: E402
from config import DEFAULT  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from io_tables import read_insole, read_storage  # noqa: E402
from subject import Subject  # noqa: E402

SUBJECT_IDS = ("AB03", "AB05", "AB07", "AB08", "AB10", "NTS01")
ACTIVITY_NAME = "single_leg_jump"
VALIDATION_ROOT = CODE_ROOT.parent / "validation" / "sensor_insole"
TRIALS_PER_SIDE = 2


def _select_trials(subject, activity, discovery):
    """First TRIALS_PER_SIDE analyzable SLJ trials per side (L, R), sorted
    by name for a deterministic, reproducible pick -- restricted to trials
    that actually have both a raw insole file and a marker force_mot file."""
    by_side = {"L": [], "R": []}
    trials = sorted(
        discovery.trials(subject, activity, "all", for_analysis=True),
        key=lambda t: t.name,
    )
    for trial in trials:
        if trial.side in by_side and trial.insole.is_file() and trial.force_mot.is_file():
            by_side[trial.side].append(trial)
    return by_side["L"][:TRIALS_PER_SIDE], by_side["R"][:TRIALS_PER_SIDE]


def _synced_series(trial):
    """Raw insole force and marker force-plate force for one trial, synced
    onto a common time base spanning the trial's FULL recorded duration
    (insole's own native time range) -- not cropped to the SLJ analysis
    window. Same peak-alignment shift convention used throughout this
    project (framework's/insole's own peak vs. the force-plate's own peak),
    reused here rather than re-derived."""
    side = trial.side
    column = "left_force" if side == "L" else "right_force"
    plate = 1 if side == "R" else 2

    insole = read_insole(trial.insole)
    force_plate = read_storage(trial.force_mot)
    insole_force = insole[column].to_numpy(float)
    marker_vertical = np.abs(force_plate[f"ground_force_{plate}_vy"].to_numpy(float))

    smoothed = np.convolve(insole_force, np.ones(7) / 7, mode="same")
    shift = float(
        force_plate["time"].iloc[int(np.argmax(marker_vertical))]
        - insole["time"].iloc[int(np.argmax(smoothed))]
    )

    time = insole["time"].to_numpy(float)  # full insole recording, uncropped
    marker_on_insole_time = np.interp(
        time + shift, force_plate["time"], marker_vertical, left=np.nan, right=np.nan
    )
    return time, insole_force, marker_on_insole_time


def collect(subject_id):
    subject = Subject(subject_id, DEFAULT)
    activity = activity_for(ACTIVITY_NAME, DEFAULT)
    discovery = TrialDiscovery(DEFAULT)
    shoe_size = subject.shoe_size

    left_trials, right_trials = _select_trials(subject, activity, discovery)
    selected = [("L", t) for t in left_trials] + [("R", t) for t in right_trials]
    if not selected:
        print(f"{subject_id}: no eligible trials (missing insole/force_mot), skipping")
        return

    out_dir = VALIDATION_ROOT / f"shoe_size_{shoe_size}" / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()
    for axis in axes:
        axis.axis("off")

    for panel, (side, trial) in enumerate(selected):
        time, insole_force, marker_force = _synced_series(trial)
        pd.DataFrame({
            "time_s": time,
            "insole_raw_N": insole_force,
            "marker_force_N": marker_force,
        }).to_csv(out_dir / f"{trial.name}_raw_vs_marker.csv", index=False)

        axis = axes[panel]
        axis.axis("on")
        axis.plot(time, insole_force, "k", label="Insole (raw)", linewidth=1)
        axis.plot(time, marker_force, "tab:red", label="Marker force plate", linewidth=1)
        axis.set_title(f"{trial.name} (side {side})")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Force (N)")
        axis.grid(True, alpha=0.3)
        if panel == 0:
            axis.legend(fontsize=8)

    figure.suptitle(
        f"{subject_id} (shoe size {shoe_size}) -- RAW insole vs. marker force plate, "
        "full trial duration, no calibration applied"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    image_path = out_dir / f"{subject_id}_raw_insole_vs_marker.png"
    figure.savefig(image_path, dpi=160)
    plt.close(figure)
    print(f"{subject_id} (shoe size {shoe_size}): {len(selected)} trials -> {image_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", help="Just this one subject instead of all 6")
    args = parser.parse_args()
    subject_ids = (args.subject,) if args.subject else SUBJECT_IDS
    for subject_id in subject_ids:
        collect(subject_id)


if __name__ == "__main__":
    main()
