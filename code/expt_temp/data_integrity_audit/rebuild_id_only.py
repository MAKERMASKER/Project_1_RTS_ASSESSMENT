"""rebuild_id_only.py -- re-run marker INVERSE DYNAMICS only, for the two
subjects whose mass was corrected.

Why ID only
-----------
`ModelScaler.scale()` derives segment GEOMETRY from the static-trial markers
and the subject's height via ScaleTool; body mass enters separately as the
setup's `<mass>` field. The mass correction changed only that field, so the
scale factors and segment geometry are unchanged, and therefore:

  * marker IK (purely kinematic -- fits model geometry to marker positions)
    is UNCHANGED and does not need re-running;
  * marker ID DOES change, because inverse dynamics uses body masses and
    inertias.

Re-running IK as well is what made the full rebuild exceed the runner's time
limit twice. This does the part that actually changed.

Prerequisite: the scaled models must already carry the corrected mass. Verify
before running -- a `cli.py marker` rerun does NOT update them (ScaleTool is
skipped unless `Config.overwrite` is True, while the file mtime still
changes, so it looks like it worked).

Run from code/:
    ..\\opensimPy\\Scripts\\python.exe expt_temp/data_integrity_audit/rebuild_id_only.py --subject AB05
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(CODE_ROOT))

from activities import activity_for  # noqa: E402
from config import DEFAULT  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from opensim_env import opensim_module  # noqa: E402
from scale_ik_id import InverseDynamicsRunner, ModelScaler  # noqa: E402
from subject import Subject  # noqa: E402

SUBJECT_IDS = ("AB03", "AB05")
ACTIVITIES = ("single_leg_jump", "lateral_hop", "skater_hop", "triple_hop")
EXPECTED_MASS = {"AB03": 89.0, "AB05": 68.3}


def _model_mass(path):
    osim = opensim_module()
    model = osim.Model(str(path))
    return sum(model.getBodySet().get(i).getMass() for i in range(model.getBodySet().getSize()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject")
    parser.add_argument("--activity")
    args = parser.parse_args()
    subject_ids = (args.subject,) if args.subject else SUBJECT_IDS
    activities = (args.activity,) if args.activity else ACTIVITIES

    config = replace(DEFAULT, overwrite=True)
    discovery = TrialDiscovery(config)
    scaler = ModelScaler(config)
    runner = InverseDynamicsRunner(config)
    done, failures = 0, []

    for subject_id in subject_ids:
        subject = Subject(subject_id, config)
        for condition in config.conditions:
            model = (scaler._models_folder(subject)
                     / f"{subject.id}_shoe_size_{subject.shoe_size}_{condition}_scaled.osim")
            if not model.is_file():
                print(f"  {subject_id}/{condition}: no scaled model -- skipping")
                continue
            mass = _model_mass(model)
            expected = EXPECTED_MASS.get(subject_id)
            status = "OK" if expected and abs(mass - expected) < 0.5 else "STALE"
            print(f"\n=== {subject_id} / {condition}: model mass {mass:.2f} kg [{status}] ===",
                  flush=True)
            if status == "STALE":
                failures.append(f"{subject_id}/{condition}: model still {mass:.2f} kg "
                                f"(expected {expected}) -- re-scale before trusting ID")
                continue
            for activity_name in activities:
                activity = activity_for(activity_name, config)
                for trial in discovery.trials(subject, activity, condition):
                    try:
                        runner.run(subject, trial, model)
                        done += 1
                    except FileNotFoundError as error:
                        print(f"  {trial.name}: skipped ({error})")
                    except Exception as error:  # noqa: BLE001
                        print(f"  {trial.name}: FAILED ({type(error).__name__}: {error})")
                        failures.append(f"{trial.name}: {error}")
                print(f"  {activity_name}: cumulative {done} ID runs", flush=True)

    print(f"\nID_REBUILD_DONE  ({done} inverse-dynamics runs)")
    print("failures:", failures if failures else "none")


if __name__ == "__main__":
    main()
