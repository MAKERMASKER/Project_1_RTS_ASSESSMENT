"""collect_insole_raw_visualization.py -- per-trial RAW bilateral force
comparison for the insole calibration investigation.

For every analyzable SLJ trial of a subject, one PNG with two panels:
  top:    force plate Right (purple) / Left (red)  vs.
          synced insole Right (light blue) / Left (orange)
  bottom: TOTAL force plate (R+L) vs. TOTAL insole (R+L), same colors

Both insole channels and both force-plate channels are read straight off
disk regardless of the trial's own side (read_insole() always returns
left_force/right_force; force_mot always has ground_force_1 [right] and
ground_force_2 [left] -- see io_tables.py / force_plates.py). RAW means no
calibration/gain is applied anywhere here, exactly like collect_insole_debug.py.

Sync: one scalar time shift per trial (insole clock vs. force-plate clock),
computed the same peak-alignment way used throughout this project -- using
the trial's own dedicated side's channel (the side actually jumped on, so
it has the cleanest, unambiguous peak) -- then applied uniformly to build
the synced timeline for BOTH channels, since the insole device's two
channels share one internal clock/offset relative to the force plate.

Output: validation/sensor_insole/RAW_visualization/<SUBJECT>/<TRIAL>.png

Run from code/, in the opensimPy environment:
    python expt_temp/collect_insole_raw_visualization.py
    python expt_temp/collect_insole_raw_visualization.py --subject AB08
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
from config import DEFAULT  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from io_tables import read_insole, read_storage  # noqa: E402
from subject import Subject  # noqa: E402

SUBJECT_IDS = ("AB03", "AB05", "AB07", "AB08", "AB10", "NTS01")
ACTIVITY_NAME = "single_leg_jump"
VALIDATION_ROOT = CODE_ROOT.parent / "validation" / "sensor_insole" / "RAW_visualization"

COLOR_PLATE_R = "purple"
COLOR_PLATE_L = "red"
COLOR_INSOLE_R = "deepskyblue"
COLOR_INSOLE_L = "orange"


def _shift_for(trial, insole, force_plate):
    """Single scalar time shift (force-plate time - insole time) from
    peak-alignment on the trial's own dedicated side -- the same
    convention used everywhere else in this project."""
    column = "left_force" if trial.side == "L" else "right_force"
    plate = 1 if trial.side == "R" else 2
    own_force = insole[column].to_numpy(float)
    own_plate = np.abs(force_plate[f"ground_force_{plate}_vy"].to_numpy(float))
    smoothed = np.convolve(own_force, np.ones(7) / 7, mode="same")
    return float(
        force_plate["time"].iloc[int(np.argmax(own_plate))]
        - insole["time"].iloc[int(np.argmax(smoothed))]
    )


def _plot_trial(trial, out_dir):
    insole = read_insole(trial.insole)
    force_plate = read_storage(trial.force_mot)
    shift = _shift_for(trial, insole, force_plate)

    time = insole["time"].to_numpy(float)
    insole_r = insole["right_force"].to_numpy(float)
    insole_l = insole["left_force"].to_numpy(float)
    plate_r_raw = np.abs(force_plate["ground_force_1_vy"].to_numpy(float))
    plate_l_raw = np.abs(force_plate["ground_force_2_vy"].to_numpy(float))
    plate_r = np.interp(time + shift, force_plate["time"], plate_r_raw, left=np.nan, right=np.nan)
    plate_l = np.interp(time + shift, force_plate["time"], plate_l_raw, left=np.nan, right=np.nan)

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    top.plot(time, plate_r, color=COLOR_PLATE_R, label="Force plate R", linewidth=1.2)
    top.plot(time, plate_l, color=COLOR_PLATE_L, label="Force plate L", linewidth=1.2)
    top.plot(time, insole_r, color=COLOR_INSOLE_R, label="Insole R (raw)", linewidth=1)
    top.plot(time, insole_l, color=COLOR_INSOLE_L, label="Insole L (raw)", linewidth=1)
    top.set_ylabel("Force (N)")
    top.set_title(f"{trial.name} (side {trial.side}) -- per-side, RAW, synced (shift={shift:.3f}s)")
    top.grid(True, alpha=0.3)
    top.legend(fontsize=8, ncol=2)

    total_plate = plate_r + plate_l
    total_insole = insole_r + insole_l
    bottom.plot(time, total_plate, color=COLOR_PLATE_R, label="Force plate TOTAL (R+L)", linewidth=1.2)
    bottom.plot(time, total_insole, color=COLOR_INSOLE_R, label="Insole TOTAL (R+L, raw)", linewidth=1.2)
    bottom.set_xlabel("Time (s)")
    bottom.set_ylabel("Force (N)")
    bottom.set_title("Total bilateral force")
    bottom.grid(True, alpha=0.3)
    bottom.legend(fontsize=8)

    figure.tight_layout()
    image_path = out_dir / f"{trial.name}.png"
    figure.savefig(image_path, dpi=150)
    plt.close(figure)
    return image_path


def collect(subject_id):
    subject = Subject(subject_id, DEFAULT)
    activity = activity_for(ACTIVITY_NAME, DEFAULT)
    discovery = TrialDiscovery(DEFAULT)

    trials = [
        trial
        for trial in discovery.trials(subject, activity, "all", for_analysis=True)
        if trial.insole.is_file() and trial.force_mot.is_file()
    ]
    if not trials:
        print(f"{subject_id}: no eligible trials (missing insole/force_mot), skipping")
        return

    out_dir = VALIDATION_ROOT / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)

    made = 0
    for trial in sorted(trials, key=lambda t: t.name):
        try:
            _plot_trial(trial, out_dir)
            made += 1
        except Exception as error:  # noqa: BLE001 -- report and keep going per trial
            print(f"  {trial.name}: FAILED ({error})")
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
