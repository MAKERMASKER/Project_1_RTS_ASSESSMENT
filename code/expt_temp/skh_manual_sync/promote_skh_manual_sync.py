"""promote_skh_manual_sync.py -- merge each <SUBJECT>_skh_manual_sync.csv
(produced by select_skh_sync.py) into that subject's subject_rules/<ID>.json,
under a new "skh_manual_sync" dict keyed by trial name:
    {"status": "synced"/"rejected", "opencap_click_s": float, "insole_click_s": float}
(shift_s is not stored -- SkaterHop._manual_sync() derives it from the two
click times directly, same convention as _finish_sync elsewhere.)

Rejected trials are also reported (not auto-added to "exclude") -- the user
promotes those into subject_rules "exclude" themselves after reviewing.

Run once per CSV (idempotent -- re-running overwrites this subject's
"skh_manual_sync" key with the CSV's current contents):
    python expt_temp/skh_manual_sync/promote_skh_manual_sync.py
"""
import csv
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from config import DEFAULT  # noqa: E402

CSV_DIR = Path(__file__).resolve().parent


def promote(csv_path: Path):
    subject_id = csv_path.name.replace("_skh_manual_sync.csv", "")
    rules_path = DEFAULT.subject_rules_root / f"{subject_id}.json"
    if not rules_path.is_file():
        print(f"  {subject_id}: no subject_rules/{subject_id}.json -- skipped")
        return

    rules = json.loads(rules_path.read_text())
    manual = {}
    rejected = []
    with csv_path.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["Status"] == "synced":
                manual[row["Trial"]] = {
                    "status": "synced",
                    "opencap_click_s": float(row["OpenCapClickTimeS"]),
                    "insole_click_s": float(row["InsoleClickTimeS"]),
                }
            elif row["Status"] == "rejected":
                manual[row["Trial"]] = {"status": "rejected"}
                rejected.append(row["Trial"])

    rules["skh_manual_sync"] = manual
    rules_path.write_text(json.dumps(rules, indent=4) + "\n")
    print(f"  {subject_id}: {len(manual)} trial(s) -> subject_rules/{subject_id}.json "
          f"({len(manual) - len(rejected)} synced, {len(rejected)} rejected)")
    if rejected:
        print(f"    rejected (review for subject_rules 'exclude'): {rejected}")


def main():
    csv_files = sorted(CSV_DIR.glob("*_skh_manual_sync.csv"))
    if not csv_files:
        raise SystemExit(f"No *_skh_manual_sync.csv files found in {CSV_DIR}")
    print(f"Promoting {len(csv_files)} manual SKH sync CSV(s):")
    for csv_path in csv_files:
        promote(csv_path)


if __name__ == "__main__":
    main()
