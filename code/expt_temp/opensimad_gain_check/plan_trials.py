"""plan_trials.py -- resolve "one NOEXO SLJ trial per subject/side" and each
one's landing-peak +/- 0.3s time window, from this project's own already-
computed framework results (peak_grf_time_s in trial_data.npz) -- no
guessing at landing times.

Run (from code/, in the opensimPy environment):
    python expt_temp/opensimad_gain_check/plan_trials.py
"""
import json
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from config import DEFAULT  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from activities import activity_for  # noqa: E402
from subject import Subject  # noqa: E402

SUBJECT_IDS = ("AB03", "AB05", "AB07", "AB08", "AB10")
HALF_WINDOW_S = 0.3
OUT_PATH = Path(__file__).resolve().parent / "trial_plan.json"


def main():
    discovery = TrialDiscovery(DEFAULT)
    activity = activity_for("single_leg_jump", DEFAULT)
    plan = []
    for subject_id in SUBJECT_IDS:
        subject = Subject(subject_id, DEFAULT)
        trials = discovery.trials(subject, activity, "NOEXO", for_analysis=True)
        by_side = {}
        for trial in trials:
            package = trial.output / "trial_data.npz"
            if not package.is_file():
                continue
            data = np.load(package, allow_pickle=True)
            if bool(data.get("quality_omit", False)):
                continue
            by_side.setdefault(trial.side, []).append((trial, float(data["peak_grf_time_s"])))

        for side, candidates in sorted(by_side.items()):
            # First (alphabetically-first trial name) qualifying candidate --
            # "one trial per subject/side", not the "best" one; any accepted,
            # non-quality-omitted trial is a fair pilot.
            trial, peak_time = sorted(candidates, key=lambda item: item[0].name)[0]
            opencap_trc = trial.effective_opencap_trc()
            if not opencap_trc or not Path(opencap_trc).is_file():
                print(f"  skip {trial.name}: no OpenCap TRC")
                continue
            session_folder = Path(opencap_trc).parents[1].name  # .../<session>/MarkerData/<name>.trc
            # The OpenSimAD repo looks up Kinematics/<trial_name>.mot by the
            # RAW OpenCap filename on disk, not this project's canonical
            # trial name -- these differ for AB03/AB07 (known "ABO3"/"AB08"
            # OpenCap-session mislabeling, see subject_rules/*.json's
            # opencap_aliases). Path(opencap_trc).stem is already the
            # alias-resolved real filename, exactly like every other
            # OpenCap file access in this project.
            opencap_trial_name = Path(opencap_trc).stem
            plan.append({
                "subject": subject_id, "side": side, "trial_name": trial.name,
                "opencap_trial_name": opencap_trial_name,
                "session_folder": session_folder,
                "peak_grf_time_s": peak_time,
                "time_window": [round(peak_time - HALF_WINDOW_S, 3), round(peak_time + HALF_WINDOW_S, 3)],
            })

    OUT_PATH.write_text(json.dumps(plan, indent=2))
    print(f"{len(plan)} trial(s) planned -> {OUT_PATH}")
    for entry in plan:
        print(f"  {entry['subject']}/{entry['side']}: {entry['trial_name']} "
              f"(opencap: {entry['opencap_trial_name']})  "
              f"session={entry['session_folder']}  window={entry['time_window']}")


if __name__ == "__main__":
    main()
