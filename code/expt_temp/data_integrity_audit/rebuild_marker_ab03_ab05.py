"""rebuild_marker_ab03_ab05.py -- force a real marker-side rebuild for the two
subjects whose mass was corrected.

Why this exists rather than `cli.py marker`
-------------------------------------------
`ModelScaler.scale()` guards its work with

    if self.config.overwrite or not anatomy.is_file():
        osim.ScaleTool(...).run()

so when the scaled model already exists the ScaleTool is SKIPPED. `cli.py`
exposes no `--overwrite` flag and `Config.overwrite` defaults to False, so a
plain `cli.py marker` rerun silently leaves the old model in place. It still
rewrites the setup XML and `_unlock_coordinates()` still rewrites the .osim,
so the file MTIME updates and the run exits 0 -- it looks like it worked.

Verified for real: after a `cli.py marker` rerun, AB03's scaled models still
carried 68.30 kg and AB05's still carried 89.00 kg, i.e. the pre-correction
(swapped) values.

This drives MarkerProcessor with `overwrite=True` so the scaling, IK and ID
actually re-run.

Run from code/:
    ..\\opensimPy\\Scripts\\python.exe expt_temp/data_integrity_audit/rebuild_marker_ab03_ab05.py
"""
import sys
from dataclasses import replace
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(CODE_ROOT))

from activities import activity_for  # noqa: E402
from config import DEFAULT  # noqa: E402
from marker_processor import MarkerProcessor  # noqa: E402
from subject import Subject  # noqa: E402

SUBJECT_IDS = ("AB03", "AB05")
ACTIVITIES = ("single_leg_jump", "lateral_hop", "skater_hop", "triple_hop")


def main():
    # Chunkable: a full two-subject rebuild exceeds the runner's background
    # time limit (measured: ~10 min got through AB03's SLJ and part of its
    # lateral hop, 20 trials). Run one subject/activity at a time.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", help="restrict to one subject")
    parser.add_argument("--activity", help="restrict to one activity")
    args = parser.parse_args()
    subject_ids = (args.subject,) if args.subject else SUBJECT_IDS
    activities = (args.activity,) if args.activity else ACTIVITIES

    config = replace(DEFAULT, overwrite=True)
    processor = MarkerProcessor(config)
    failures = []
    for subject_id in subject_ids:
        subject = Subject(subject_id, config)
        print(f"\n######## {subject_id}  (metadata mass = {subject.mass_kg} kg) ########",
              flush=True)
        for activity_name in activities:
            activity = activity_for(activity_name, config)
            print(f"=== {subject_id} / {activity_name} ===", flush=True)
            try:
                # C3D markers and plate forces are unchanged by a mass fix.
                processor.run(subject, activity, "all", convert_c3d=False)
            except Exception as error:  # noqa: BLE001 -- keep going, report at the end
                print(f"  FAILED ({type(error).__name__}: {error})")
                failures.append(f"{subject_id}/{activity_name}: {error}")
    print("\nREBUILD_DONE")
    print("failures:", failures if failures else "none")


if __name__ == "__main__":
    main()
