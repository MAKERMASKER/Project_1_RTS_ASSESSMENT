"""foot_frame_sync.py -- OpenCap contact detection built on the STANDARD
framework foot anatomical frame, A/B-tested against the toe-height detector.

Why
---
`auto_sync.py`'s OpenCap detector used `InsoleGeometry.foot_height()`, which
averages ONLY the `toe1` and `toe5` markers:

    np.nanmean(np.c_[marker(...,"toe1")[:,1], marker(...,"toe5")[:,1]], axis=1)

The heel is not in it. So at heel-strike (heel down, toes up) it reads
airborne while the foot is already loaded, and at push-off (heel lifted, toes
still down) it reads in-contact after unloading has begun. Both skew the
detected stance late, and neither is visible in a binary-train plot -- which is
why the contacts looked unreliable.

That function was written as an SLJ toe-height sync diagnostic, not as a
contact detector, so this is a misuse rather than a bug in it.

This detector instead uses the same foot anatomical frame the framework
pipeline already builds for the COP transform
(`InsoleGeometry.frames()` -> origin + [ap, ml, vertical] rotation):

1. Project BOTH ends of the foot onto its own sole plane by offsetting along
   the foot's OWN vertical axis (which rotates with the foot, so the geometry
   stays right through heel-strike and toe-off).
2. Sole height = the LOWEST of those sole points. Whatever part of the foot is
   actually down defines contact -- heel at landing, forefoot at push-off.
3. Ground = a low percentile of that signal, per trial, so it self-calibrates
   and never depends on OpenCap's vertical origin. (A constant sole offset
   largely cancels between signal and reference; it matters only through the
   rotation, which is exactly the part worth getting right.)
4. Foot pitch, from the angle between the foot's own vertical axis and global
   vertical, is recorded so heel-down vs toe-down is visible in the plots.

The state machine, histogram matching and validation are unchanged and
imported from `auto_sync.py`, so this is a clean A/B on the detector alone.

Plots now show the CONTINUOUS signals (sole height, pitch, speed) under the
detected states, not just binary trains -- the previous plots made it
impossible to see why a contact was called where it was.

Run from code/:
    ..\\opensimPy\\Scripts\\python.exe expt_temp/state_machine_sync/foot_frame_sync.py
    ..\\opensimPy\\Scripts\\python.exe expt_temp/state_machine_sync/foot_frame_sync.py --subject AB08
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent.parent
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(HERE))

from activities import activity_for  # noqa: E402
from config import DEFAULT  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from insole_geometry import InsoleGeometry  # noqa: E402
from io_tables import read_insole, read_trc  # noqa: E402
from subject import Subject  # noqa: E402

from auto_sync import (  # noqa: E402
    ENTER_HEIGHT_M, EXIT_HEIGHT_M, GROUND_PERCENTILE, MAX_PLANTED_SPEED_MPS,
    SUBJECT_IDS, _apply_dwell, _hysteretic, _insole_contacts, _match_offset,
    _opencap_contacts, _reference_offset,
)

ACTIVITIES = ("lateral_hop", "skater_hop")
RESULTS = HERE / "results_foot_frame"

# Markers sit above the sole; offsetting along the foot's OWN vertical axis
# keeps the sole estimate correct while the foot pitches.
SOLE_OFFSET_M = 0.03


def _foot_frame_signals(trc, side):
    """(sole_height_above_ground, pitch_deg, speed) on trc's own time base."""
    time = np.asarray(trc["time"], float)
    dt = float(np.median(np.diff(time)))
    heel = InsoleGeometry.marker(trc, time, side, "heel")
    toe1 = InsoleGeometry.marker(trc, time, side, "toe1")
    toe5 = InsoleGeometry.marker(trc, time, side, "toe5")
    _origin, rotation = InsoleGeometry.frames(trc, time, side)
    foot_vertical = rotation[:, :, 2]

    heel_sole = heel - SOLE_OFFSET_M * foot_vertical
    toe_sole = 0.5 * (toe1 + toe5) - SOLE_OFFSET_M * foot_vertical
    lowest = np.minimum(heel_sole[:, 1], toe_sole[:, 1])

    finite = np.isfinite(lowest)
    ground = np.nanpercentile(lowest[finite], GROUND_PERCENTILE) if finite.any() else np.nan
    above = lowest - ground

    pitch = np.degrees(np.arccos(np.clip(foot_vertical[:, 1], -1.0, 1.0)))

    centre = (heel + toe1 + toe5) / 3.0
    horizontal = np.column_stack([centre[:, 0], centre[:, 2]])
    speed = np.r_[0.0, np.linalg.norm(np.diff(horizontal, axis=0), axis=1) / dt]
    speed = np.convolve(np.nan_to_num(speed), np.ones(5) / 5, mode="same")
    return above, pitch, speed


def _opencap_contacts_foot_frame(trc, side):
    time = np.asarray(trc["time"], float)
    dt = float(np.median(np.diff(time)))
    above, _pitch, speed = _foot_frame_signals(trc, side)
    raw = _hysteretic(-above, -ENTER_HEIGHT_M, -EXIT_HEIGHT_M)
    raw &= speed <= MAX_PLANTED_SPEED_MPS
    mask, stances, _standing = _apply_dwell(raw, dt)
    return mask, np.array([time[s] for s, _e in stances])


# Height is a marginal observable for these hops: the sole only reaches
# 8-13 cm in flight, barely clearing a 3.5/7 cm threshold band, so adjacent
# hops merge. Foot PITCH separates far better -- ~0-5 deg planted against
# 35-40 deg in flight -- because in a lateral hop the foot rotates much more
# decisively than it rises.
ENTER_PITCH_DEG = 12.0
EXIT_PITCH_DEG = 22.0


def _opencap_contacts_pitch(trc, side):
    time = np.asarray(trc["time"], float)
    dt = float(np.median(np.diff(time)))
    _above, pitch, speed = _foot_frame_signals(trc, side)
    raw = _hysteretic(-pitch, -ENTER_PITCH_DEG, -EXIT_PITCH_DEG)
    raw &= speed <= MAX_PLANTED_SPEED_MPS
    mask, stances, _standing = _apply_dwell(raw, dt)
    return mask, np.array([time[s] for s, _e in stances])


def _plot(trial, side, trc, insole_time, insole_mask, offset, reference,
          spread, matched, mask_new, mask_old):
    out = RESULTS / "plots" / trial.name.split("_")[0]
    out.mkdir(parents=True, exist_ok=True)
    time = np.asarray(trc["time"], float)
    above, pitch, speed = _foot_frame_signals(trc, side)

    figure, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)

    axes[0].plot(time, 100 * above, color="darkorange", lw=1.0, label="sole height above ground (cm)")
    axes[0].axhline(100 * ENTER_HEIGHT_M, color="green", ls="--", lw=0.8, label="enter")
    axes[0].axhline(100 * EXIT_HEIGHT_M, color="red", ls="--", lw=0.8, label="exit")
    axes[0].set_ylim(-2, 40)
    axes[0].set_ylabel("sole height (cm)")
    axes[0].legend(fontsize=7, loc="upper right")

    axes[1].plot(time, pitch, color="purple", lw=1.0, label="foot pitch from vertical (deg)")
    axes[1].plot(time, speed, color="teal", lw=0.9, label="horizontal foot speed (m/s)")
    axes[1].axhline(MAX_PLANTED_SPEED_MPS, color="teal", ls=":", lw=0.8)
    axes[1].set_ylabel("pitch (deg) / speed")
    axes[1].legend(fontsize=7, loc="upper right")

    axes[2].fill_between(insole_time + offset, 0, 1, where=insole_mask, step="mid",
                         color="deepskyblue", alpha=0.6, label="insole contact (shifted)")
    axes[2].fill_between(time, 1.1, 2.1, where=mask_new, step="mid",
                         color="darkorange", alpha=0.6, label="OpenCap contact (foot frame)")
    axes[2].fill_between(time, 2.2, 3.2, where=mask_old, step="mid",
                         color="grey", alpha=0.5, label="OpenCap contact (toe height, old)")
    axes[2].set_yticks([0.5, 1.6, 2.7])
    axes[2].set_yticklabels(["insole", "foot frame", "toe height"])
    axes[2].set_xlabel("Time (s, OpenCap clock)")
    axes[2].legend(fontsize=7, loc="upper right")
    for axis in axes:
        axis.grid(True, alpha=0.3)

    error = "" if not np.isfinite(reference) else f" | manual={reference:.3f}s err={offset - reference:+.3f}s"
    figure.suptitle(f"{trial.name} [{side}] foot-frame offset={offset:.3f}s "
                    f"spread={spread:.3f}s matched={matched}{error}")
    figure.tight_layout()
    figure.savefig(out / f"{trial.name}_{side}_footframe.png", dpi=130)
    plt.close(figure)


def _analyse(trial, subject, activity):
    trc = read_trc(trial.effective_opencap_trc())
    insole = read_insole(trial.insole)
    insole_time = insole.time.to_numpy(float)
    sides = ("R", "L") if trial.side is None else (trial.side,)

    rows = []
    for side in sides:
        column = "right_force" if side == "R" else "left_force"
        insole_mask, insole_events, _ = _insole_contacts(insole[column].to_numpy(float), insole_time)
        mask_new, events_new = _opencap_contacts_foot_frame(trc, side)
        mask_old, events_old, _ = _opencap_contacts(trc, side)
        mask_pitch, events_pitch = _opencap_contacts_pitch(trc, side)

        offset, spread, matched, slope = _match_offset(insole_events, events_new)
        old_offset, old_spread, old_matched, _ = _match_offset(insole_events, events_old)
        pitch_offset, pitch_spread, pitch_matched, _ = _match_offset(insole_events, events_pitch)
        reference = _reference_offset(subject.rules, activity.name, trial.name)

        _plot(trial, side, trc, insole_time, insole_mask, offset, reference,
              spread, matched, mask_new, mask_old)

        rows.append({
            "Subject": subject.id, "Activity": activity.name, "Trial": trial.name, "Side": side,
            "InsoleEvents": len(insole_events),
            "FootFrameEvents": len(events_new), "ToeHeightEvents": len(events_old),
            "PitchEvents": len(events_pitch),
            "FootFrameMatched": matched, "ToeHeightMatched": old_matched,
            "PitchMatched": pitch_matched,
            "PitchOffset_s": round(pitch_offset, 4) if np.isfinite(pitch_offset) else np.nan,
            "PitchError_s": round(pitch_offset - reference, 4) if np.isfinite(pitch_offset) and np.isfinite(reference) else np.nan,
            "PitchSpread_s": round(pitch_spread, 4) if np.isfinite(pitch_spread) else np.nan,
            "FootFrameOffset_s": round(offset, 4) if np.isfinite(offset) else np.nan,
            "ToeHeightOffset_s": round(old_offset, 4) if np.isfinite(old_offset) else np.nan,
            "ManualOffset_s": round(reference, 4) if np.isfinite(reference) else np.nan,
            "FootFrameError_s": round(offset - reference, 4) if np.isfinite(offset) and np.isfinite(reference) else np.nan,
            "ToeHeightError_s": round(old_offset - reference, 4) if np.isfinite(old_offset) and np.isfinite(reference) else np.nan,
            "FootFrameSpread_s": round(spread, 4) if np.isfinite(spread) else np.nan,
            "ToeHeightSpread_s": round(old_spread, 4) if np.isfinite(old_spread) else np.nan,
            "DriftSlope": round(slope, 6) if np.isfinite(slope) else np.nan,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject")
    args = parser.parse_args()
    subject_ids = (args.subject,) if args.subject else SUBJECT_IDS

    discovery = TrialDiscovery(DEFAULT)
    rows = []
    for subject_id in subject_ids:
        subject = Subject(subject_id, DEFAULT)
        for activity_name in ACTIVITIES:
            activity = activity_for(activity_name, DEFAULT)
            trials = [
                t for t in discovery.trials(subject, activity, "all", for_analysis=True)
                if t.insole.is_file() and t.effective_opencap_trc()
                and Path(t.effective_opencap_trc()).is_file()
            ]
            for trial in sorted(trials, key=lambda t: t.name):
                try:
                    rows.extend(_analyse(trial, subject, activity))
                except Exception as error:  # noqa: BLE001
                    print(f"  {trial.name}: FAILED ({type(error).__name__}: {error})")
            print(f"{subject_id}/{activity_name}: {len(trials)} trials")

    RESULTS.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "foot_frame_vs_toe_height.csv", index=False)
    print(f"\n{len(table)} (trial, side) results -> {RESULTS / 'foot_frame_vs_toe_height.csv'}")

    scored = table[np.isfinite(table.FootFrameError_s) & np.isfinite(table.ToeHeightError_s)]
    if not len(scored):
        return
    columns = [("toe height", "ToeHeightError_s", "ToeHeightSpread_s", "ToeHeightEvents"),
               ("foot frame", "FootFrameError_s", "FootFrameSpread_s", "FootFrameEvents"),
               ("pitch", "PitchError_s", "PitchSpread_s", "PitchEvents")]
    print(f"\n=== A/B against {len(scored)} hand-synced references ===")
    print(f"{'':<24}" + "".join(f"{label:>13}" for label, *_ in columns))
    print(f"{'median |error| (s)':<24}"
          + "".join(f"{scored[e].abs().median():>13.3f}" for _l, e, _s, _n in columns))
    for bound in (0.05, 0.10, 0.25, 1.00):
        print(f"{'within ' + format(bound, '.2f') + ' s':<24}"
              + "".join(f"{int((scored[e].abs() <= bound).sum()):>13}" for _l, e, _s, _n in columns))
    print(f"{'median spread (s)':<24}"
          + "".join(f"{scored[s].median():>13.3f}" for _l, _e, s, _n in columns))
    print(f"{'median events found':<24}"
          + "".join(f"{scored[n].median():>13.0f}" for _l, _e, _s, n in columns))
    print(f"{'median insole events':<24}{scored.InsoleEvents.median():>13.0f}")
    print(f"\nevent-count gap vs insole (median): "
          + ", ".join(f"{label} {(scored[n] - scored.InsoleEvents).median():+.0f}"
                      for label, _e, _s, n in columns))


if __name__ == "__main__":
    main()
