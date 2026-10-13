"""apply_ideal_gains.py -- write per-activity IDEAL gains into each subject's
calibration file, so the framework pipeline can be run in a benchmark
configuration.

    --apply     write the ideal gains (backs up each file first)
    --restore   put the backups back
    (no flag)   dry run: print what would change

**These gains are derived from force-plate ground truth.** They are NOT
field-deployable -- they need the very measurement the framework pipeline
exists to replace. Writing them in produces a CEILING/BENCHMARK run ("how good
can this pipeline be if the gain is perfect"), never evidence of the markerless
pipeline's independent accuracy. Every value written carries a
`vertical_force_scale_source` saying so, so a later reader cannot mistake it
for a deployable calibration, and the original subject-wide gain is preserved
untouched under the "L"/"R" keys.

Per-trial fits come from fit_ideal_gains.py (run that first); the value used
per subject/side/activity is the MEDIAN across that group's trials.

Run from code/:
    ..\\opensimPy\\Scripts\\python.exe expt_temp/cross_activity_gain_check/apply_ideal_gains.py
    ..\\opensimPy\\Scripts\\python.exe expt_temp/cross_activity_gain_check/apply_ideal_gains.py --apply
    ..\\opensimPy\\Scripts\\python.exe expt_temp/cross_activity_gain_check/apply_ideal_gains.py --restore
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(CODE_ROOT))

from config import DEFAULT  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
SOURCE_TAG = "ideal_least_squares_vs_force_plate__GROUND_TRUTH_DERIVED__NOT_FIELD_DEPLOYABLE"


def _calibration_path(subject_id):
    return DEFAULT.calibration_root / f"{subject_id}_insole_marker_foot_calibration.json"


def _backup_path(subject_id):
    return DEFAULT.calibration_root / f"{subject_id}_insole_marker_foot_calibration.prebenchmark.json"


def restore():
    restored = 0
    for backup in sorted(DEFAULT.calibration_root.glob("*_insole_marker_foot_calibration.prebenchmark.json")):
        target = Path(str(backup).replace(".prebenchmark", ""))
        shutil.copy2(backup, target)
        backup.unlink()
        print(f"restored {target.name}")
        restored += 1
    print(f"{restored} calibration file(s) restored" if restored else "no backups found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="actually write the gains")
    parser.add_argument("--restore", action="store_true", help="restore pre-benchmark calibrations")
    args = parser.parse_args()

    if args.restore:
        restore()
        return

    table = pd.read_csv(RESULTS / "ideal_gains_per_trial.csv")
    grouped = table.groupby(["Subject", "Activity", "Side"]).agg(
        IdealGain=("IdealGain", "median"), Trials=("IdealGain", "size")
    ).reset_index()

    for subject_id, subject_rows in grouped.groupby("Subject"):
        path = _calibration_path(subject_id)
        if not path.is_file():
            print(f"{subject_id}: no calibration file -- skipping")
            continue
        calibration = json.loads(path.read_text())
        per_activity = {}
        for _, row in subject_rows.iterrows():
            entry = per_activity.setdefault(row.Activity, {})
            entry[row.Side] = {
                "vertical_force_scale": round(float(row.IdealGain), 4),
                "vertical_force_scale_source": SOURCE_TAG,
                "fit_from_trials": int(row.Trials),
            }
            subject_wide = calibration[row.Side]["vertical_force_scale"]
            print(f"{subject_id:<7} {row.Activity:<17} {row.Side}  "
                  f"{subject_wide:.4f} -> {row.IdealGain:.4f}  (n={int(row.Trials)})")

        if not args.apply:
            continue
        backup = _backup_path(subject_id)
        if not backup.is_file():
            shutil.copy2(path, backup)
        calibration["per_activity"] = per_activity
        calibration["per_activity_note"] = (
            "Ground-truth (force-plate) derived ideal gains -- BENCHMARK ONLY, not "
            "field-deployable. The subject-wide 'L'/'R' vertical_force_scale above is "
            "the real, deployable stance-fit value and is left untouched. See "
            "code/expt_temp/cross_activity_gain_check/README.md."
        )
        path.write_text(json.dumps(calibration, indent=2))

    print("\nDRY RUN -- pass --apply to write." if not args.apply else "\nWritten.")


if __name__ == "__main__":
    main()
