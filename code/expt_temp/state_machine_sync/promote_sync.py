"""promote_sync.py -- write the final per-trial sync offsets into
subject_rules so the framework uses them.

Offsets go into the EXISTING `lh_manual_sync` / `skh_manual_sync` keys, in the
existing `{status, opencap_click_s, insole_click_s}` shape. Both activities
already prefer that dict over their automatic anchor detection, so no
production code changes -- the framework simply picks the new offsets up.

An automatic offset has no literal click pair, so one is synthesised:
`insole_click_s` is the first detected insole contact (a real loading event,
which matters because `_manual_sync` refines to the local force peak within
+/-0.5 s of that click) and `opencap_click_s = insole_click_s + offset`.
Manual decisions keep the human's own clicks verbatim.

Refuses to promote unless every trial carries a sync, since a partial
promotion would silently leave some trials on the old, unreliable automatic
anchor while others use the new one -- the two are not comparable.

    --apply    write (backs up each subject_rules file first)
    --restore  put the backups back
    (default)  dry run

Run from code/:
    ..\\opensimPy\\Scripts\\python.exe expt_temp/state_machine_sync/promote_sync.py
    ..\\opensimPy\\Scripts\\python.exe expt_temp/state_machine_sync/promote_sync.py --apply
"""
import argparse
import json
import shutil
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

from auto_sync import _insole_contacts  # noqa: E402
from validate_sync import _final_offsets  # noqa: E402

RULES_ROOT = DEFAULT.subject_rules_root
KEY = {"lateral_hop": "lh_manual_sync", "skater_hop": "skh_manual_sync"}
SOURCE_TAG = "state_machine_auto_sync"


# The anchor must leave room for the activity's own analysis window around
# it (LateralHop._manual_sync uses (-1.0 s, +0.3 s)), or _finish_sync clips to
# an empty overlap.
ANCHOR_MARGIN_BEFORE_S = 1.5
ANCHOR_MARGIN_AFTER_S = 0.5


def _anchor_click(trial, offset):
    """An insole contact that lands INSIDE the OpenCap capture once shifted.

    Taking the first detected contact is wrong: the insole records
    continuously and typically starts well before OpenCap, so its first
    contact often maps to a NEGATIVE OpenCap time. `_finish_sync` then clips
    its window to nothing and the trial dies with "No synchronized overlap" --
    which is exactly what happened to 8 trials on the first promotion.
    """
    insole = read_insole(trial.insole)
    time = insole.time.to_numpy(float)
    side = trial.side if trial.side else "R"
    column = "right_force" if side == "R" else "left_force"
    _mask, events, _standing = _insole_contacts(insole[column].to_numpy(float), time)

    trc = read_trc(trial.effective_opencap_trc())
    low = float(np.min(trc["time"])) + ANCHOR_MARGIN_BEFORE_S
    high = float(np.max(trc["time"])) - ANCHOR_MARGIN_AFTER_S
    for event in events:
        if low <= event + offset <= high:
            return float(event)
    # No contact maps inside the capture -- fall back to its midpoint, which
    # is always valid even if it is not a loading event.
    return float((low + high) / 2 - offset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="promote even with trials missing a sync (not advised)")
    args = parser.parse_args()

    if args.restore:
        restored = 0
        for backup in sorted(RULES_ROOT.glob("*.presync.json")):
            target = backup.with_name(backup.name.replace(".presync", ""))
            shutil.copy2(backup, target)
            backup.unlink()
            print(f"restored {target.name}")
            restored += 1
        print(f"{restored} file(s) restored" if restored else "no backups found")
        return

    offsets, decisions = _final_offsets()
    discovery = TrialDiscovery(DEFAULT)
    per_subject, missing = {}, []

    for subject_id in sorted(decisions.Subject.unique()):
        subject = Subject(subject_id, DEFAULT)
        for activity_name in ("lateral_hop", "skater_hop"):
            activity = activity_for(activity_name, DEFAULT)
            for trial in sorted(discovery.trials(subject, activity, "all", for_analysis=True),
                                key=lambda t: t.name):
                if not (trial.insole.is_file() and trial.effective_opencap_trc()
                        and Path(trial.effective_opencap_trc()).is_file()):
                    continue
                offset, source = offsets.get(trial.name, (np.nan, "MISSING"))
                entry = per_subject.setdefault(subject_id, {}).setdefault(KEY[activity_name], {})
                if source == "MISSING":
                    missing.append((subject_id, activity_name, trial.name))
                    continue
                if source == "rejected":
                    entry[trial.name] = {"status": "rejected", "source": SOURCE_TAG}
                    continue
                insole_click = _anchor_click(trial, offset)
                entry[trial.name] = {
                    "status": "synced",
                    "insole_click_s": round(insole_click, 4),
                    "opencap_click_s": round(insole_click + offset, 4),
                    "source": SOURCE_TAG if source == "auto" else "manual_zoom_gui",
                }

    total = sum(len(v) for subject in per_subject.values() for v in subject.values())
    print(f"{total} trial(s) resolved, {len(missing)} missing")
    for subject_id, groups in sorted(per_subject.items()):
        for key, entries in sorted(groups.items()):
            sources = pd.Series([e.get("source") for e in entries.values()]).value_counts().to_dict()
            print(f"  {subject_id:<7} {key:<16} {len(entries):>3} trial(s)  {sources}")

    if missing and not args.force:
        print("\nREFUSING to promote -- these trials have no sync yet:")
        for subject_id, activity_name, trial_name in missing:
            print(f"  {subject_id:<7} {activity_name:<13} {trial_name}")
        print("\nRun select_sync_zoom.py for the flagged subjects first "
              "(or pass --force, not advised).")
        return

    if not args.apply:
        print("\nDRY RUN -- pass --apply to write.")
        return

    for subject_id, groups in sorted(per_subject.items()):
        path = RULES_ROOT / f"{subject_id}.json"
        if not path.is_file():
            print(f"  {subject_id}: no rules file -- skipping")
            continue
        backup = path.with_name(f"{subject_id}.presync.json")
        if not backup.is_file():
            shutil.copy2(path, backup)
        rules = json.loads(path.read_text())
        for key, entries in groups.items():
            # MERGE, never replace: a trial excluded via trial_status.csv is
            # no longer returned by discovery, so it is absent from `entries`.
            # A wholesale replace would silently delete its rejection record
            # and quietly resurrect the trial on some later run.
            existing = dict(rules.get(key, {}))
            existing.update(entries)
            rules[key] = existing
        path.write_text(json.dumps(rules, indent=2))
        print(f"  wrote {path.name}")
    print("\nWritten. The framework will use these on its next run.")


if __name__ == "__main__":
    main()
