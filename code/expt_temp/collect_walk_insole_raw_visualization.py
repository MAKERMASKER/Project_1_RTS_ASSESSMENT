"""collect_walk_insole_raw_visualization.py -- per-trial RAW bilateral force
comparison for WALK trials, same idea and plot style as
collect_insole_raw_visualization.py (SLJ), adapted for gait:

  top:    force plate Right (purple) / Left (red)  vs.
          synced insole Right (light blue) / Left (orange)
  bottom: TOTAL force plate (R+L) vs. TOTAL insole (R+L), same colors

RAW means no calibration/gain is applied anywhere here, exactly like the SLJ
version. Force-plate right/left come from ForcePlateProcessor's already-
combined resultant (Walk.marker_plates(): right = plates 2+4, left = plates
3+1+6+5, force-weighted-averaged where a foot spans multiple plates -- see
force_plates.py -- so this is already the correct per-foot signal, not a
single raw plate).

Sync is different from the SLJ version: a WALK trial has no single dominant
side/peak to align on (Trial.side is None -- see activities.py's Walk class
and Activity.has_side_token), so instead of peak-alignment this uses
cross-correlation between the TOTAL (both feet summed) insole and force-
plate signals -- the standard, more general technique for aligning two
periodic/multi-peak signals when there's no one unambiguous landmark.

Subjects: AB03, AB05, AB07, AB10 (AB08 has WALK c3d but zero WALK insole
files; NTS01 has no WALK data at all -- both skipped).

Output: validation/walking/sensor_insole/RAW_visualization/<SUBJECT>/<TRIAL>.png

Run from code/, in the opensimPy environment:
    python expt_temp/collect_walk_insole_raw_visualization.py
    python expt_temp/collect_walk_insole_raw_visualization.py --subject AB03
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from activities import activity_for  # noqa: E402
from config import DEFAULT  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from io_tables import read_insole, read_storage  # noqa: E402
from subject import Subject  # noqa: E402

SUBJECT_IDS = ("AB03", "AB05", "AB07", "AB10")
ACTIVITY_NAME = "walk"
VALIDATION_ROOT = CODE_ROOT.parent / "validation" / "walking" / "sensor_insole" / "RAW_visualization"

COLOR_PLATE_R = "purple"
COLOR_PLATE_L = "red"
COLOR_INSOLE_R = "deepskyblue"
COLOR_INSOLE_L = "orange"

# Trials omitted entirely -- not a sync-algorithm problem, the underlying
# data is unusable: AB05_NOEXO_WALK_1's insole recording is only 3.1s long
# against an 8.4s trial and never overlaps the plate-active window at all
# (confirmed: no valid shift produces any plate/insole correlation).
# AB10_NOEXO_WALK_1 gives an ambiguous, low-confidence plate<->insole L/R
# correlation inconsistent with its own WALK_2 trial (same subject, same
# plate layout) -- flagged and omitted per the user's call rather than
# guessed at further.
OMITTED_TRIALS = {
    "AB05_NOEXO_WALK_1": "insole recording (3.1s) does not overlap the plate-active window (8.4s trial)",
    "AB10_NOEXO_WALK_1": "ambiguous/low-confidence plate<->insole side correlation, inconsistent with WALK_2",
}

# Ground truth: which foot's step was first to land in each trial (told by
# the user, who watched the capture) -- used only to VALIDATE the sync
# below, not to compute it. AB03 omitted: no ground truth given for it.
FIRST_STEP_SIDE = {
    "AB05_NOEXO_WALK_1": "L", "AB05_NOEXO_WALK_2": "R",
    "AB10_NOEXO_WALK_1": "L", "AB10_NOEXO_WALK_2": "R",
    "AB07_NOEXO_WALK_1": "L", "AB07_NOEXO_WALK_2": "R",
}
SYNC_PRIOR_MARGIN_S = 0.75


def _select_trials(subject, activity, discovery):
    trials = [
        trial
        for trial in discovery.trials(subject, activity, "NOEXO", for_analysis=True)
        if trial.insole.is_file() and trial.force_mot.is_file() and trial.name not in OMITTED_TRIALS
    ]
    return sorted(trials, key=lambda t: t.name)


def _cross_correlation_shift(insole_time, insole_total, plate_time, plate_total):
    """Time to ADD to insole_time so it aligns with plate_time -- found by
    cross-correlating the two TOTAL (both-feet) force signals, constrained
    to a small window around a strong PRIOR instead of an unconstrained
    global search.

    Gait is periodic (~1-2s stride), so an unconstrained cross-correlation
    can lock onto the wrong stride -- a spurious lag one full step-cycle off
    that still "looks" plausible (alternating pulses roughly overlapping)
    but is actually wrong. The insole and mocap captures were started at
    close to the same real moment for every WALK trial, so the true shift
    must be within SYNC_PRIOR_MARGIN_S of `prior = plate_time[0] -
    insole_time[0]` (mapping insole's own first sample directly onto the
    force plate's t=0) -- restricting the search to that window removes the
    stride-period ambiguity entirely rather than trying to resolve it after
    the fact."""
    dt = float(np.median(np.diff(insole_time)))
    plate_uniform_time = np.arange(plate_time.min(), plate_time.max(), dt)
    plate_uniform = np.interp(plate_uniform_time, plate_time, plate_total)

    insole_centered = insole_total - np.mean(insole_total)
    plate_centered = plate_uniform - np.mean(plate_uniform)

    correlation = np.correlate(plate_centered, insole_centered, mode="full")
    center_index = len(insole_centered) - 1  # index where lag_samples == 0
    margin_samples = int(round(SYNC_PRIOR_MARGIN_S / dt))
    low = max(0, center_index - margin_samples)
    high = min(len(correlation) - 1, center_index + margin_samples)
    best_index = low + int(np.argmax(correlation[low:high + 1]))
    lag_samples = best_index - center_index

    prior_shift = plate_uniform_time[0] - insole_time[0]
    return prior_shift + lag_samples * dt


def _first_step_side(time, insole_r, insole_l, plate_r, plate_l, contact_n=50.0, peak_prominence=150.0):
    """(plate_first_side, insole_first_side) -- which foot's signal rises
    first, read independently off each sensor's own synced data, purely for
    validating the sync direction/quality against the user's own observed
    ground truth (see FIRST_STEP_SIDE). Not used to compute the sync."""
    def first_crossing(force):
        active = np.flatnonzero(force >= contact_n)
        return time[active[0]] if len(active) else np.inf

    plate_r_on, plate_l_on = first_crossing(plate_r), first_crossing(plate_l)
    plate_first = None if not np.isfinite(min(plate_r_on, plate_l_on)) else ("R" if plate_r_on < plate_l_on else "L")

    window_start = min(plate_r_on, plate_l_on) - 0.5
    window = (time >= window_start) & (time <= window_start + 3.5) if np.isfinite(window_start) else np.zeros_like(time, dtype=bool)
    wt = time[window]

    def first_peak_time(force):
        peaks, _ = find_peaks(force[window], prominence=peak_prominence)
        return wt[peaks[0]] if len(peaks) else np.inf

    r_peak, l_peak = first_peak_time(insole_r), first_peak_time(insole_l)
    insole_first = None if not np.isfinite(min(r_peak, l_peak)) else ("R" if r_peak < l_peak else "L")
    return plate_first, insole_first


def _trial_series(trial):
    insole = read_insole(trial.insole)
    force_plate = read_storage(trial.force_mot)

    insole_total = insole["right_force"].to_numpy(float) + insole["left_force"].to_numpy(float)
    plate_r_raw = np.abs(force_plate["ground_force_1_vy"].to_numpy(float))
    plate_l_raw = np.abs(force_plate["ground_force_2_vy"].to_numpy(float))
    plate_total = plate_r_raw + plate_l_raw

    shift = _cross_correlation_shift(
        insole["time"].to_numpy(float), insole_total,
        force_plate["time"].to_numpy(float), plate_total,
    )

    time = insole["time"].to_numpy(float)
    insole_r = insole["right_force"].to_numpy(float)
    insole_l = insole["left_force"].to_numpy(float)
    plate_r = np.interp(time + shift, force_plate["time"], plate_r_raw, left=np.nan, right=np.nan)
    plate_l = np.interp(time + shift, force_plate["time"], plate_l_raw, left=np.nan, right=np.nan)
    return {
        "time": time, "insole_r": insole_r, "insole_l": insole_l,
        "plate_r": plate_r, "plate_l": plate_l, "shift": shift,
    }


def collect(subject_id):
    subject = Subject(subject_id, DEFAULT)
    activity = activity_for(ACTIVITY_NAME, DEFAULT)
    discovery = TrialDiscovery(DEFAULT)

    trials = _select_trials(subject, activity, discovery)
    if not trials:
        print(f"{subject_id}: no eligible WALK trials (missing insole/force_mot), skipping")
        return

    out_dir = VALIDATION_ROOT / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)

    made = 0
    for trial in trials:
        try:
            series = _trial_series(trial)
        except Exception as error:  # noqa: BLE001 -- report and keep going per trial
            print(f"  {trial.name}: FAILED ({error})")
            continue

        plate_first, insole_first = _first_step_side(
            series["time"], series["insole_r"], series["insole_l"],
            np.nan_to_num(series["plate_r"]), np.nan_to_num(series["plate_l"]),
        )
        expected_first = FIRST_STEP_SIDE.get(trial.name)
        if expected_first:
            status = "OK" if plate_first == insole_first == expected_first else "MISMATCH"
            check = f"first step: expected {expected_first}, plate={plate_first}, insole={insole_first} [{status}]"
            print(f"  {trial.name}: {check}")
        else:
            check = f"first step (no ground truth given): plate={plate_first}, insole={insole_first}"

        figure, (top, bottom) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
        top.plot(series["time"], series["plate_r"], color=COLOR_PLATE_R, label="Force plate R", linewidth=1.2)
        top.plot(series["time"], series["plate_l"], color=COLOR_PLATE_L, label="Force plate L", linewidth=1.2)
        top.plot(series["time"], series["insole_r"], color=COLOR_INSOLE_R, label="Insole R (raw)", linewidth=1)
        top.plot(series["time"], series["insole_l"], color=COLOR_INSOLE_L, label="Insole L (raw)", linewidth=1)
        top.set_ylabel("Force (N)")
        top.set_title(
            f"{trial.name} -- RAW, cross-correlation synced (shift={series['shift']:.3f}s)\n{check}"
        )
        top.grid(True, alpha=0.3)
        top.legend(fontsize=8, ncol=2)

        total_plate = series["plate_r"] + series["plate_l"]
        total_insole = series["insole_r"] + series["insole_l"]
        bottom.plot(series["time"], total_plate, color=COLOR_PLATE_R, label="Force plate TOTAL (R+L)", linewidth=1.2)
        bottom.plot(series["time"], total_insole, color=COLOR_INSOLE_R, label="Insole TOTAL (R+L, raw)", linewidth=1.2)
        bottom.set_xlabel("Time (s)")
        bottom.set_ylabel("Force (N)")
        bottom.set_title("Total bilateral force")
        bottom.grid(True, alpha=0.3)
        bottom.legend(fontsize=8)

        figure.tight_layout()
        image_path = out_dir / f"{trial.name}.png"
        figure.savefig(image_path, dpi=150)
        plt.close(figure)
        made += 1
    print(f"{subject_id}: {made}/{len(trials)} trials -> {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", help="Just this one subject instead of all 4")
    args = parser.parse_args()
    subject_ids = (args.subject,) if args.subject else SUBJECT_IDS
    for subject_id in subject_ids:
        collect(subject_id)


if __name__ == "__main__":
    main()
