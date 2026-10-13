"""run_filter_validate.py -- batch-run AutomaticTrcFilter + TrcRepairValidator
(trc_repair.py) across every eligible Skater Hop / Lateral Hop trial, for
review before any repair-promote decision.

This deliberately stops short of RepairPromoter -- nothing here touches
subject_rules/<ID>.json or PROCESSED/OPENCAP_REPAIRED/, so no production
output changes as a result of running this script. It exists purely to
generate the candidate filtered TRC + comparison plot/RMS summary for every
trial in one pass, instead of the user running cli.py's repair-filter/
repair-validate one trial at a time. Promotion (if any trial's repair looks
like a genuine improvement) stays an explicit, separate, per-trial human
decision via `cli.py repair-promote --subject <ID> --activity <activity>
--trial <name>`.

Scope: AB03/AB05/AB07/AB08/AB10, skater_hop + lateral_hop. NTS01 excluded
per the user's standing instruction (not being processed for now).

Output (per trial, same layout cli.py's repair-filter/repair-validate would
produce): results/<SUBJECT>/<condition>/<activity>/<trial>/trc_repair/ and
.../trc_repair_validation/. Plus one combined summary CSV here:
expt_temp/trc_repair_batch/<activity>_filter_validate_summary.csv
"""
import sys
import traceback
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from activities import activity_for  # noqa: E402
from config import DEFAULT  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from subject import Subject  # noqa: E402
from trc_repair import AutomaticTrcFilter, TrcRepairValidator  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
SUBJECT_IDS = ("AB03", "AB05", "AB07", "AB08", "AB10")
ACTIVITY_NAMES = ("skater_hop", "lateral_hop")


def _side_for(trial):
    """SkaterHop is bilateral (trial.side is None) but always validates on
    the fixed sync side, "R" -- see SkaterHop.synchronize()'s sync_side="R"
    in activities.py. LateralHop/SingleLegJump have a real trial.side."""
    return trial.side if trial.side is not None else "R"


def main():
    discovery = TrialDiscovery(DEFAULT)
    trc_filter = AutomaticTrcFilter()
    validator = TrcRepairValidator(DEFAULT)

    for activity_name in ACTIVITY_NAMES:
        activity = activity_for(activity_name, DEFAULT)
        rows = []
        for subject_id in SUBJECT_IDS:
            subject = Subject(subject_id, DEFAULT)
            trials = discovery.trials(subject, activity, "all", for_analysis=True)
            for trial in trials:
                opencap_trc = trial.effective_opencap_trc()
                if not opencap_trc or not Path(opencap_trc).is_file():
                    print(f"  {trial.name}: skipped (no OpenCap TRC)")
                    continue
                print(f"{trial.name}: filter + validate")
                repair_dir = trial.output / "trc_repair"
                try:
                    filter_result = trc_filter.process_trial(trial, repair_dir)
                except ValueError as error:
                    print(f"  skipped (filter: {error})")
                    rows.append({"Trial": trial.name, "Subject": subject_id, "Status": f"filter_failed: {error}"})
                    continue
                try:
                    summary = validator.validate(
                        subject, activity, trial, filter_result["filtered_trc"], side=_side_for(trial)
                    )
                except Exception as error:  # noqa: BLE001 -- report, keep batch going
                    print(f"  skipped (validate: {error})")
                    traceback.print_exc()
                    rows.append({"Trial": trial.name, "Subject": subject_id, "Status": f"validate_failed: {error}"})
                    continue

                for _, joint_row in summary.iterrows():
                    rows.append({
                        "Trial": trial.name,
                        "Subject": subject_id,
                        "Status": "ok",
                        "MarkersFlaggedForReview": filter_result["markers_flagged_for_review"],
                        **joint_row.to_dict(),
                    })
                print(f"  markers flagged: {filter_result['markers_flagged_for_review']}")

        table = pd.DataFrame(rows)
        out_path = OUT_DIR / f"{activity_name}_filter_validate_summary.csv"
        table.to_csv(out_path, index=False)
        print(f"{activity_name}: {len(table)} row(s) -> {out_path}")


if __name__ == "__main__":
    main()
