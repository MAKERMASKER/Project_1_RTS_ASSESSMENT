"""auto_sync.py -- automatic insole<->OpenCap synchronisation via a per-foot
contact state machine and event-train matching.

The problem
-----------
Insole and OpenCap each start at an arbitrary time; ONE constant scalar offset
per trial relates them (no drift). The force plate must NOT participate --
it is validation, not pipeline. So the offset has to come from insole and
OpenCap alone.

Today's production methods anchor on a SINGLE event, which is one observation
of one unknown with no redundancy and no way to detect its own failure. It was
unreliable enough that every LH and SKH trial was hand-synced instead
(`lh_manual_sync` / `skh_manual_sync` in subject_rules). A 30-50 cycle bout
offers 30-50 independent observations of that same scalar; using all of them
is a completely different, over-determined problem.

The method
----------
1. Reduce BOTH modalities to the same discrete per-foot contact signal.
   - Insole: force is a direct contact measurement.
   - OpenCap: no force, so contact is inferred from foot kinematics --
     foot height near the ground AND low horizontal foot speed. The ground
     reference is taken per trial as a low percentile of foot height, so it
     self-calibrates and never depends on OpenCap's absolute vertical origin.

2. Clean each with the SAME two-stage state machine (FLIGHT <-> STANCE):
   - hysteresis (enter and exit thresholds differ) kills boundary chatter;
   - minimum dwell rejects single-sample glitches;
   - maximum stance duration reclassifies a long hold as STANDING, which
     never enters the event train. That is what structurally prevents the
     already-documented failure of mistaking a static pre-trial stance for
     the first hop.
   MIN_STANCE_S / MAX_STANCE_S are LateralHop's own validated durations
   (0.15 s / 1.8 s), not new guesses.

3. Match the two touchdown trains. Every pair (insole i, OpenCap j) votes for
   an offset `s_j - t_i`; the true offset is the sharp mode of that histogram,
   because correct pairs agree while spurious pairs scatter. This is immune to
   unequal lead-in (the insole records continuously, OpenCap is trimmed), which
   is precisely what breaks first-event anchoring. The mode is then refined to
   the median over matched pairs.

4. Self-validate. The SPREAD of matched residuals is a graded confidence
   measure, replacing today's binary quality_omit. A linear TREND in those
   residuals would indicate clock drift -- checked for free, and expected to
   be absent.

Validation
----------
Against the 52 hand-synced LH/SKH trials, whose clicked
(opencap_click_s, insole_click_s) pairs give a reference offset of
`opencap_click_s - insole_click_s` (exactly what `_finish_sync` derives).
Reproducing those means the manual work can be automated.

Run from code/:
    ..\\opensimPy\\Scripts\\python.exe expt_temp/state_machine_sync/auto_sync.py
    ..\\opensimPy\\Scripts\\python.exe expt_temp/state_machine_sync/auto_sync.py --subject AB08
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(CODE_ROOT))

from activities import activity_for  # noqa: E402
from config import DEFAULT  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from insole_geometry import InsoleGeometry  # noqa: E402
from io_tables import read_insole, read_trc  # noqa: E402
from subject import Subject  # noqa: E402

SUBJECT_IDS = ("AB03", "AB05", "AB07", "AB08", "AB10", "NTS01")
ACTIVITIES = ("lateral_hop", "skater_hop")
RESULTS = Path(__file__).resolve().parent / "results"

# LateralHop's own validated hop-duration signature.
MIN_STANCE_S = 0.15
MAX_STANCE_S = 1.8
MIN_FLIGHT_S = 0.05

# OpenCap contact inference. Ground is a low percentile of foot height, so
# these are heights ABOVE that reference, in metres.
GROUND_PERCENTILE = 5.0
ENTER_HEIGHT_M = 0.035          # below this -> touchdown
EXIT_HEIGHT_M = 0.070           # above this -> liftoff (hysteresis gap)
MAX_PLANTED_SPEED_MPS = 1.2     # horizontal foot speed permitted during stance

MATCH_BIN_S = 0.04
MATCH_TOLERANCE_S = 0.15


def _hysteretic(score, enter_above, exit_below):
    """Boolean contact from a score where HIGHER means more like contact."""
    mask = np.zeros(len(score), dtype=bool)
    state = False
    for index, value in enumerate(score):
        if not np.isfinite(value):
            mask[index] = False
            continue
        if not state and value >= enter_above:
            state = True
        elif state and value <= exit_below:
            state = False
        mask[index] = state
    return mask


def _runs(mask):
    out, start = [], None
    for index, value in enumerate(np.append(np.asarray(mask, bool), False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            out.append((start, index - 1))
            start = None
    return out


def _apply_dwell(mask, dt):
    """Drop too-short stances, bridge too-short flights, and reclassify an
    over-long stance as STANDING (excluded, not merged)."""
    mask = np.asarray(mask, bool).copy()
    for start, end in _runs(~mask):                       # bridge brief flights
        if start > 0 and end < len(mask) - 1 and (end - start + 1) * dt < MIN_FLIGHT_S:
            mask[start:end + 1] = True
    stances, standing = [], []
    for start, end in _runs(mask):
        duration = (end - start + 1) * dt
        if duration < MIN_STANCE_S:
            mask[start:end + 1] = False                   # glitch
        elif duration > MAX_STANCE_S:
            mask[start:end + 1] = False                   # STANDING, not a hop
            standing.append((start, end))
        else:
            stances.append((start, end))
    return mask, stances, standing


def _insole_contacts(force, time):
    dt = float(np.median(np.diff(time)))
    threshold = float(DEFAULT.contact_force_n)
    raw = _hysteretic(np.asarray(force, float), threshold, 0.5 * threshold)
    mask, stances, standing = _apply_dwell(raw, dt)
    return mask, np.array([time[s] for s, _e in stances]), standing


def _opencap_contacts(trc, side):
    time = np.asarray(trc["time"], float)
    dt = float(np.median(np.diff(time)))
    height = InsoleGeometry.foot_height(trc, side)
    ground = np.nanpercentile(height[np.isfinite(height)], GROUND_PERCENTILE)
    above = height - ground

    centre = InsoleGeometry.foot_center(trc, side)
    horizontal = np.column_stack([centre[:, 0], centre[:, 2]])
    speed = np.r_[0.0, np.linalg.norm(np.diff(horizontal, axis=0), axis=1) / dt]
    speed = np.convolve(np.nan_to_num(speed), np.ones(5) / 5, mode="same")

    # score high == more like contact: low height, and planted.
    score = -above
    raw = _hysteretic(score, -ENTER_HEIGHT_M, -EXIT_HEIGHT_M)
    raw &= speed <= MAX_PLANTED_SPEED_MPS
    mask, stances, standing = _apply_dwell(raw, dt)
    return mask, np.array([time[s] for s, _e in stances]), standing


def _match_offset(insole_events, opencap_events):
    """(offset, residual_spread_s, matched, drift_slope) by histogram voting."""
    if len(insole_events) < 2 or len(opencap_events) < 2:
        return np.nan, np.nan, 0, np.nan
    differences = (opencap_events[:, None] - insole_events[None, :]).ravel()
    edges = np.arange(differences.min(), differences.max() + MATCH_BIN_S, MATCH_BIN_S)
    if len(edges) < 2:
        return np.nan, np.nan, 0, np.nan
    counts, edges = np.histogram(differences, bins=edges)
    coarse = float(edges[int(np.argmax(counts))] + MATCH_BIN_S / 2)

    residuals, positions = [], []
    for event in opencap_events:
        gaps = event - (insole_events + coarse)
        nearest = int(np.argmin(np.abs(gaps)))
        if abs(gaps[nearest]) <= MATCH_TOLERANCE_S:
            residuals.append(event - insole_events[nearest])
            positions.append(event)
    if len(residuals) < 2:
        return np.nan, np.nan, 0, np.nan
    residuals = np.array(residuals)
    offset = float(np.median(residuals))
    spread = float(np.percentile(residuals, 75) - np.percentile(residuals, 25))
    slope = float(np.polyfit(np.array(positions), residuals, 1)[0]) if len(residuals) >= 3 else np.nan
    return offset, spread, len(residuals), slope


def _reference_offset(rules, activity_name, trial_name):
    key = "lh_manual_sync" if activity_name == "lateral_hop" else "skh_manual_sync"
    manual = rules.get(key, {}).get(trial_name)
    if not manual or manual.get("status") != "synced":
        return np.nan
    return float(manual["opencap_click_s"]) - float(manual["insole_click_s"])


def _plot(trial, activity, side, insole_time, insole_mask, opencap_time,
          opencap_mask, offset, reference, spread, matched):
    out = RESULTS / "plots" / activity.name / trial.name.split("_")[0]
    out.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(15, 3.4))
    axis.fill_between(insole_time + offset, 0, 1, where=insole_mask,
                      color="deepskyblue", alpha=0.55, step="mid", label="insole contact (shifted)")
    axis.fill_between(opencap_time, 1.1, 2.1, where=opencap_mask,
                      color="darkorange", alpha=0.55, step="mid", label="OpenCap contact (kinematic)")
    axis.set_yticks([0.5, 1.6])
    axis.set_yticklabels(["insole", "OpenCap"])
    axis.set_xlabel("Time (s, OpenCap clock)")
    error = "" if not np.isfinite(reference) else f" | manual={reference:.3f}s err={offset - reference:+.3f}s"
    axis.set_title(f"{trial.name} [{side}] auto offset={offset:.3f}s "
                   f"spread={spread:.3f}s matched={matched}{error}")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=7, loc="upper right")
    figure.tight_layout()
    figure.savefig(out / f"{trial.name}_{side}_states.png", dpi=130)
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
        opencap_mask, opencap_events, _ = _opencap_contacts(trc, side)
        offset, spread, matched, slope = _match_offset(insole_events, opencap_events)
        reference = _reference_offset(subject.rules, activity.name, trial.name)
        _plot(trial, activity, side, insole_time, insole_mask,
              np.asarray(trc["time"], float), opencap_mask, offset, reference, spread, matched)
        rows.append({
            "Subject": subject.id, "Activity": activity.name, "Trial": trial.name, "Side": side,
            "InsoleEvents": len(insole_events), "OpenCapEvents": len(opencap_events),
            "Matched": matched,
            "AutoOffset_s": round(offset, 4) if np.isfinite(offset) else np.nan,
            "ManualOffset_s": round(reference, 4) if np.isfinite(reference) else np.nan,
            "Error_s": round(offset - reference, 4) if np.isfinite(offset) and np.isfinite(reference) else np.nan,
            "ResidualSpread_s": round(spread, 4) if np.isfinite(spread) else np.nan,
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
    table.to_csv(RESULTS / "auto_sync.csv", index=False)
    print(f"\n{len(table)} (trial, side) results -> {RESULTS / 'auto_sync.csv'}")

    scored = table[np.isfinite(table.Error_s)]
    if len(scored):
        absolute = scored.Error_s.abs()
        print(f"\n=== vs {len(scored)} hand-synced references ===")
        print(f"  median |error| : {absolute.median():.3f} s")
        print(f"  within 0.10 s  : {(absolute <= 0.10).sum()}/{len(scored)}")
        print(f"  within 0.25 s  : {(absolute <= 0.25).sum()}/{len(scored)}")
        print(f"  within 1.00 s  : {(absolute <= 1.00).sum()}/{len(scored)}")
        print(f"  median residual spread: {scored.ResidualSpread_s.median():.3f} s")


if __name__ == "__main__":
    main()
