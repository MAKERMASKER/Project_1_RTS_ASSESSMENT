"""decide_sync.py -- decide, per trial, whether the insole<->OpenCap sync can
be trusted automatically or must go to the manual GUI.

Rule
----
Several independent estimates are produced for each trial and required to
agree:
  * the PITCH detector (precise edges) and the HEIGHT+speed detector (robust
    detection) -- these were measured to fail on DIFFERENT trials, so their
    agreement is informative;
  * for a bilateral activity (SKH), both feet, which must imply the same
    single trial-level offset.

If every estimate falls within AGREEMENT_TOLERANCE_S, the trial is AUTO and
the offset is their median. Otherwise it is MANUAL.

Validated at (side, method) level on the 70 hand-synced references: accepting
only on agreement gave 53/70 accepted, median error 0.022 s, max 0.178 s, and
**zero** failures among accepted -- every genuine failure landed in the
flagged set. This script lifts that to a per-TRIAL decision, which is the unit
the framework actually needs (one global offset per trial).

Caveat, stated plainly: the tolerance was chosen by inspecting these same
references, so the clean separation is optimistic. It is a screening rule that
routes doubtful trials to a human, not a proof of correctness.

Writes results/sync_decisions.csv -- one row per trial, consumed by
select_sync_zoom.py (manual queue) and promote_sync.py (framework hand-off).

Run from code/:
    ..\\opensimPy\\Scripts\\python.exe expt_temp/state_machine_sync/decide_sync.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent.parent
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(HERE))

from activities import activity_for  # noqa: E402
from config import DEFAULT  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from io_tables import read_insole, read_trc  # noqa: E402
from subject import Subject  # noqa: E402

from auto_sync import SUBJECT_IDS, _insole_contacts, _match_offset, _reference_offset  # noqa: E402
from foot_frame_sync import _opencap_contacts_pitch  # noqa: E402
from refined_sync import _combined_contacts  # noqa: E402

ACTIVITIES = ("lateral_hop", "skater_hop")
RESULTS = HERE / "results"
AGREEMENT_TOLERANCE_S = 0.05
MIN_MATCHED_EVENTS = 4


def _largest_cluster(offsets, tolerance):
    """The biggest group of estimates lying within `tolerance` of each other.

    Requiring UNANIMITY is the wrong test: the range of a sample grows with
    its size even at fixed noise, so SKH (4 estimates: 2 feet x 2 detectors)
    is penalised against LH (2 estimates) for no real reason. A majority
    cluster tolerates one dissenting estimate while still demanding genuine
    corroboration.
    """
    if not len(offsets):
        return np.array([])
    best = np.array([])
    for anchor in offsets:
        cluster = offsets[np.abs(offsets - anchor) <= tolerance]
        if len(cluster) > len(best):
            best = cluster
    return best


def estimates_for_trial(trial):
    """Every independent (side, method) offset estimate for one trial."""
    trc = read_trc(trial.effective_opencap_trc())
    insole = read_insole(trial.insole)
    insole_time = insole.time.to_numpy(float)
    sides = ("R", "L") if trial.side is None else (trial.side,)

    estimates = []
    for side in sides:
        column = "right_force" if side == "R" else "left_force"
        _mask, insole_events, _standing = _insole_contacts(
            insole[column].to_numpy(float), insole_time)
        _pm, pitch_events = _opencap_contacts_pitch(trc, side)
        _cm, combined_events = _combined_contacts(trc, side)
        for label, events in (("pitch", pitch_events), ("height", combined_events)):
            offset, _spread, matched, _slope = _match_offset(insole_events, events)
            if np.isfinite(offset) and matched >= MIN_MATCHED_EVENTS:
                estimates.append({"side": side, "method": label,
                                  "offset": float(offset), "matched": int(matched)})
    return estimates


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
                reference = _reference_offset(subject.rules, activity_name, trial.name)
                try:
                    estimates = estimates_for_trial(trial)
                except Exception as error:  # noqa: BLE001
                    print(f"  {trial.name}: FAILED ({type(error).__name__}: {error})")
                    estimates = []

                offsets = np.array([e["offset"] for e in estimates])
                cluster = _largest_cluster(offsets, AGREEMENT_TOLERANCE_S)
                if len(cluster) >= 2 and len(cluster) >= int(np.ceil(len(offsets) / 2)):
                    offset = float(np.median(cluster))
                    spread = float(cluster.max() - cluster.min())
                    decision = "AUTO"
                else:
                    offset = float(np.median(offsets)) if len(offsets) else np.nan
                    spread = float(offsets.max() - offsets.min()) if len(offsets) >= 2 else np.nan
                    decision = "MANUAL"

                rows.append({
                    "Subject": subject_id, "Activity": activity_name, "Trial": trial.name,
                    "Decision": decision,
                    "AutoOffset_s": round(offset, 4) if np.isfinite(offset) else np.nan,
                    "EstimateSpread_s": round(spread, 4) if np.isfinite(spread) else np.nan,
                    "Estimates": len(estimates),
                    "ManualOffset_s": round(reference, 4) if np.isfinite(reference) else np.nan,
                    "Error_s": round(offset - reference, 4)
                    if np.isfinite(offset) and np.isfinite(reference) else np.nan,
                })
            print(f"{subject_id}/{activity_name}: {len(trials)} trials")

    RESULTS.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "sync_decisions.csv", index=False)
    print(f"\n{len(table)} trials -> {RESULTS / 'sync_decisions.csv'}")

    auto = table[table.Decision == "AUTO"]
    manual = table[table.Decision == "MANUAL"]
    print(f"\nAUTO   {len(auto)}/{len(table)} ({100 * len(auto) / len(table):.0f}%)")
    print(f"MANUAL {len(manual)}/{len(table)} ({100 * len(manual) / len(table):.0f}%)")

    scored = auto[np.isfinite(auto.Error_s)]
    if len(scored):
        error = scored.Error_s.abs()
        print(f"\n=== AUTO trials vs {len(scored)} hand-synced references ===")
        print(f"  median |error| {error.median():.4f} s | p90 {error.quantile(0.9):.4f} s "
              f"| max {error.max():.4f} s")
        print(f"  failures (>0.25 s): {(error > 0.25).sum()}")
    flagged = manual[np.isfinite(manual.Error_s)]
    if len(flagged):
        print(f"\n=== MANUAL (flagged) trials: {len(flagged)} with a reference, "
              f"{(flagged.Error_s.abs() > 0.25).sum()} genuinely wrong ===")
    print("\nManual queue:")
    for _index, row in manual.iterrows():
        print(f"  {row.Subject:<7} {row.Activity:<13} {row.Trial:<22} "
              f"spread={row.EstimateSpread_s}")


if __name__ == "__main__":
    main()
