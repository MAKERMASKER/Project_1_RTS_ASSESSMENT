"""rerun_framework.py -- reprocess LH and SKH framework output with the
promoted state-machine syncs.

Calls `RtsAssessmentProject.run_framework()` DIRECTLY and deliberately does
not go through `cli.py framework`, because that chains straight into
`run_analysis()` -- the SLJ-style one-row-per-trial analysis, which is not a
valid analysis definition for these multi-cycle activities and whose output
was already discarded once. Per-cycle analysis is `segment_hops.py`'s job and
is run separately afterwards.

Run from code/:
    ..\\opensimPy\\Scripts\\python.exe expt_temp/state_machine_sync/rerun_framework.py
"""
import sys
import traceback
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(CODE_ROOT))

from project import RtsAssessmentProject  # noqa: E402

SUBJECT_IDS = ("AB03", "AB05", "AB07", "AB08", "AB10", "NTS01")
ACTIVITIES = ("lateral_hop", "skater_hop")


def main():
    failures = []
    for activity in ACTIVITIES:
        for subject_id in SUBJECT_IDS:
            print(f"=== {subject_id} / {activity} ===", flush=True)
            try:
                RtsAssessmentProject(subject_id).run_framework(activity, "all", None)
            except Exception as error:  # noqa: BLE001 -- one subject must not stop the batch
                print(f"  FAILED ({type(error).__name__}: {error})")
                traceback.print_exc()
                failures.append(f"{subject_id}/{activity}")
    print("\nRERUN_FRAMEWORK_DONE")
    print("failures:", failures if failures else "none")


if __name__ == "__main__":
    main()
