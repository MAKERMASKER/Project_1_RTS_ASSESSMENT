"""refined_sync.py -- maximum-precision insole<->OpenCap sync over the
overlapping duration.

Two independent fixes, composed.

1. Detection vs edge location are different jobs, so use a different signal
   for each.
   - **Detect** stances on sole height + horizontal speed. Height is the
     robust observable: it knows the foot is near the ground, and it finds the
     most events.
   - **Locate the edge** with foot pitch. Pitch has a far larger dynamic range
     for these hops (~0-5 deg planted vs 35-40 deg in flight) than height does
     (8-13 cm of flight clearance against a 3.5/7 cm band), so the crossing
     instant is sharply defined. Measured alone, pitch was 2.5x more accurate
     (0.028 s vs 0.070 s median) and 3.2x more often within 50 ms, but missed
     more events -- exactly the split of strengths this exploits.
   Each height-detected touchdown is snapped to the nearest pitch
   down-crossing within a short window; if there is none, the height edge is
   kept unchanged.

2. **Duration fingerprint** to break the periodic degeneracy. A lateral-hop
   train is 30-50 near-evenly-spaced hops, so the offset histogram has several
   near-equal peaks one hop period apart, and taking the tallest sometimes
   lands on a neighbour -- every previous gross failure was off by exactly one
   period (0.53, 0.60, 1.00, 1.16 s against a 0.6-1.0 s period). Touchdown
   TIMES cannot resolve this; inter-touchdown INTERVALS can, because real hops
   are not identical. So the top-K histogram peaks are each scored by how well
   the matched interval sequences agree, and the best-agreeing candidate wins.
   Intervals are used rather than stance durations because they are edge
   differences, so any systematic detection bias cancels.

Note the event-count deficit against the insole is expected and benign:
OpenCap stops recording before the insole does. Only the overlapping duration
is being synced, and only precision over that overlap is optimised.

Validated against the 70 hand-synced LH/SKH references.

Run from code/:
    ..\\opensimPy\\Scripts\\python.exe expt_temp/state_machine_sync/refined_sync.py
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

from auto_sync import (  # noqa: E402
    ENTER_HEIGHT_M, EXIT_HEIGHT_M, MATCH_BIN_S, MATCH_TOLERANCE_S,
    MAX_PLANTED_SPEED_MPS, SUBJECT_IDS, _apply_dwell, _hysteretic,
    _insole_contacts, _match_offset, _reference_offset,
)
from foot_frame_sync import (  # noqa: E402
    ENTER_PITCH_DEG, _foot_frame_signals, _opencap_contacts_pitch,
)

ACTIVITIES = ("lateral_hop", "skater_hop")
RESULTS = HERE / "results_refined"

EDGE_REFINE_WINDOW_S = 0.20
TOP_K_CANDIDATES = 6
CANDIDATE_SEPARATION_S = 0.25


def _combined_contacts(trc, side):
    """Detect on height+speed, then snap each touchdown to the pitch edge."""
    time = np.asarray(trc["time"], float)
    dt = float(np.median(np.diff(time)))
    above, pitch, speed = _foot_frame_signals(trc, side)

    raw = _hysteretic(-above, -ENTER_HEIGHT_M, -EXIT_HEIGHT_M)
    raw &= speed <= MAX_PLANTED_SPEED_MPS
    mask, stances, _standing = _apply_dwell(raw, dt)

    window = max(1, int(round(EDGE_REFINE_WINDOW_S / dt)))
    events = []
    for start, _end in stances:
        low = max(0, start - window)
        high = min(len(pitch) - 1, start + window)
        segment = pitch[low:high + 1]
        below = np.flatnonzero(np.isfinite(segment) & (segment <= ENTER_PITCH_DEG))
        if len(below):
            # first moment the foot is flat within the window == touchdown
            events.append(time[low + int(below[0])])
        else:
            events.append(time[start])
    return mask, np.array(events)


def _match_with_fingerprint(insole_events, opencap_events):
    """(offset, spread, matched, interval_error, candidates_considered).

    Scores the top-K offset-histogram peaks by inter-touchdown interval
    agreement instead of blindly taking the tallest.
    """
    if len(insole_events) < 3 or len(opencap_events) < 3:
        return np.nan, np.nan, 0, np.nan, 0
    differences = (opencap_events[:, None] - insole_events[None, :]).ravel()
    edges = np.arange(differences.min(), differences.max() + MATCH_BIN_S, MATCH_BIN_S)
    if len(edges) < 2:
        return np.nan, np.nan, 0, np.nan, 0
    counts, edges = np.histogram(differences, bins=edges)

    candidates, chosen = [], []
    for index in np.argsort(counts)[::-1]:
        if counts[index] < 3:
            break
        centre = float(edges[index] + MATCH_BIN_S / 2)
        if any(abs(centre - other) < CANDIDATE_SEPARATION_S for other in chosen):
            continue
        chosen.append(centre)
        candidates.append(centre)
        if len(candidates) >= TOP_K_CANDIDATES:
            break

    # Score candidates by MATCH COUNT first, interval agreement only as a
    # tie-break. Ranking on mean interval error alone is wrong: a candidate
    # that matches 4 events by luck posts a lower mean error than the correct
    # one matching 40 events with honest noise, so the sparse impostor wins.
    # Match count is what separates a gross (multi-second) miss; the
    # fingerprint only needs to separate the near-ties one hop period apart,
    # which have almost identical counts.
    records = []
    for candidate in candidates:
        insole_index, opencap_index = [], []
        for j, event in enumerate(opencap_events):
            gaps = event - (insole_events + candidate)
            nearest = int(np.argmin(np.abs(gaps)))
            if abs(gaps[nearest]) <= MATCH_TOLERANCE_S:
                insole_index.append(nearest)
                opencap_index.append(j)
        if len(insole_index) < 4 or len(set(insole_index)) < 4:
            continue
        insole_index = np.array(insole_index)
        opencap_index = np.array(opencap_index)
        insole_intervals = np.diff(insole_events[insole_index])
        opencap_intervals = np.diff(opencap_events[opencap_index])
        keep = (insole_intervals > 0) & (opencap_intervals > 0)
        if keep.sum() < 3:
            continue
        interval_error = float(np.mean(np.abs(insole_intervals[keep] - opencap_intervals[keep])))
        residuals = opencap_events[opencap_index] - insole_events[insole_index]
        records.append({
            "matched": len(residuals), "interval_error": interval_error,
            "offset": float(np.median(residuals)),
            "spread": float(np.percentile(residuals, 75) - np.percentile(residuals, 25)),
        })
    if not records:
        return np.nan, np.nan, 0, np.nan, len(candidates)
    best_count = max(record["matched"] for record in records)
    contenders = [record for record in records if record["matched"] >= 0.9 * best_count]
    best = min(contenders, key=lambda record: record["interval_error"])
    return (best["offset"], best["spread"], best["matched"],
            best["interval_error"], len(candidates))


def _analyse(trial, subject, activity):
    trc = read_trc(trial.effective_opencap_trc())
    insole = read_insole(trial.insole)
    insole_time = insole.time.to_numpy(float)
    sides = ("R", "L") if trial.side is None else (trial.side,)

    rows = []
    for side in sides:
        column = "right_force" if side == "R" else "left_force"
        _m, insole_events, _s = _insole_contacts(insole[column].to_numpy(float), insole_time)
        _mask, combined_events = _combined_contacts(trc, side)
        _pm, pitch_events = _opencap_contacts_pitch(trc, side)
        reference = _reference_offset(subject.rules, activity.name, trial.name)

        offset, spread, matched, interval_error, candidates = _match_with_fingerprint(
            insole_events, combined_events)
        plain_offset, _ps, _pmatched, _slope = _match_offset(insole_events, combined_events)
        pitch_offset, _s2, _m2, _s3 = _match_offset(insole_events, pitch_events)

        rows.append({
            "Subject": subject.id, "Activity": activity.name, "Trial": trial.name, "Side": side,
            "InsoleEvents": len(insole_events), "CombinedEvents": len(combined_events),
            "Matched": matched, "Candidates": candidates,
            "IntervalError_s": round(interval_error, 4) if np.isfinite(interval_error) else np.nan,
            "ManualOffset_s": round(reference, 4) if np.isfinite(reference) else np.nan,
            "RefinedOffset_s": round(offset, 4) if np.isfinite(offset) else np.nan,
            "RefinedError_s": round(offset - reference, 4) if np.isfinite(offset) and np.isfinite(reference) else np.nan,
            "CombinedPlainError_s": round(plain_offset - reference, 4) if np.isfinite(plain_offset) and np.isfinite(reference) else np.nan,
            "PitchOnlyError_s": round(pitch_offset - reference, 4) if np.isfinite(pitch_offset) and np.isfinite(reference) else np.nan,
            "Spread_s": round(spread, 4) if np.isfinite(spread) else np.nan,
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
    table.to_csv(RESULTS / "refined_sync.csv", index=False)
    print(f"\n{len(table)} (trial, side) results -> {RESULTS / 'refined_sync.csv'}")

    scored = table[np.isfinite(table.RefinedError_s)]
    if not len(scored):
        return
    variants = [("pitch only", "PitchOnlyError_s"),
                ("combined, plain match", "CombinedPlainError_s"),
                ("combined + fingerprint", "RefinedError_s")]
    print(f"\n=== against {len(scored)} hand-synced references ===")
    print(f"{'':<26}" + "".join(f"{label:>24}" for label, _c in variants))
    print(f"{'median |error| (s)':<26}"
          + "".join(f"{scored[c].abs().median():>24.3f}" for _l, c in variants))
    for bound in (0.02, 0.05, 0.10, 0.25, 1.00):
        print(f"{'within ' + format(bound, '.2f') + ' s':<26}"
              + "".join(f"{int((scored[c].abs() <= bound).sum()):>24}" for _l, c in variants))
    print(f"\n=== by activity, combined + fingerprint ===")
    print(scored.groupby("Activity").RefinedError_s.apply(
        lambda v: pd.Series({"n": len(v), "median_abs": round(v.abs().median(), 3),
                             "max_abs": round(v.abs().max(), 3)})).unstack().to_string())


if __name__ == "__main__":
    main()
