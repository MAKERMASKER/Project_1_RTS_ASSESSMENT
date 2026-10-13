"""plan_pushoff_windows.py -- for every trial in trial_plan.json, replace
the landing-peak +/- 0.3s window with a push-off (jump-initiation) +/- 0.3s
window: the moment each trial's own already-computed framework contact
mask goes False for the first time (liftoff into flight, "insole goes to
zero"), taken from the SAME trial_data.npz already used for the landing
window -- no new detection logic, just a different anchor point in an
already-computed signal.

Run (from code/, in the opensimPy environment):
    python expt_temp/opensimad_gain_check/plan_pushoff_windows.py
"""
import json
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from config import DEFAULT  # noqa: E402

PLAN_PATH = Path(__file__).resolve().parent / "trial_plan.json"
OUT_PATH = Path(__file__).resolve().parent / "trial_plan_pushoff.json"
HALF_WINDOW_S = 0.3


def liftoff_time(trial_name: str, condition: str) -> float:
    npz_path = DEFAULT.results_root / trial_name.split("_")[0] / "framework" / "single_leg_jump" / condition.lower() / trial_name / "trial_data.npz"
    data = np.load(npz_path, allow_pickle=True)
    time = data["time"]
    contact = data["contact"].astype(bool)
    changes = np.diff(np.r_[False, contact, False].astype(int))
    ends = np.flatnonzero(changes == -1) - 1
    starts = np.flatnonzero(changes == 1)
    peak_index = int(np.argmax(data["force"]))
    # The liftoff immediately preceding the recorded landing peak -- the
    # last contact-segment END that occurs strictly before the peak (the
    # segment containing the peak itself starts before it too, but its own
    # END is the recorded window's tail, not a real liftoff -- must exclude it).
    preceding = [end for start, end in zip(starts, ends) if end < peak_index]
    if not preceding:
        raise ValueError(f"{trial_name}: no contact segment found before the landing peak")
    return float(time[preceding[-1]])


def main():
    plan = json.loads(PLAN_PATH.read_text())
    out = []
    for entry in plan:
        condition = "NOEXO"  # this whole plan is NOEXO-only per the user's request
        try:
            liftoff = liftoff_time(entry["trial_name"], condition)
        except (FileNotFoundError, ValueError) as error:
            print(f"  skip {entry['trial_name']}: {error}")
            continue
        new_entry = dict(entry)
        new_entry["liftoff_time_s"] = round(liftoff, 3)
        new_entry["time_window"] = [round(liftoff - HALF_WINDOW_S, 3), round(liftoff + HALF_WINDOW_S, 3)]
        out.append(new_entry)

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"{len(out)} trial(s) planned -> {OUT_PATH}")
    for entry in out:
        print(f"  {entry['subject']}/{entry['side']}: {entry['trial_name']} "
              f"(opencap: {entry['opencap_trial_name']})  window={entry['time_window']}")


if __name__ == "__main__":
    main()
