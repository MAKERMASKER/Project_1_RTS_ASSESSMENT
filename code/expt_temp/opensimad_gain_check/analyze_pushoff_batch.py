"""analyze_pushoff_batch.py -- for every planned push-off trial with a GRF
result (converged or non-convergent), compare OpenSimAD's peak vertical
GRF against the marker ground truth's own peak vertical force during the
SAME push-off window (force plate, shifted onto the framework clock via
the same peak-matching heuristic used throughout this session), and
compute the OpenSimAD-implied insole gain vs. the raw insole reading.

Run (from code/, in the opensimPy environment):
    python expt_temp/opensimad_gain_check/analyze_pushoff_batch.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from config import DEFAULT  # noqa: E402
from io_tables import read_storage, read_insole  # noqa: E402
from subject import Subject  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from activities import activity_for  # noqa: E402

PLAN_PATH = Path(__file__).resolve().parent / "trial_plan_pushoff.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def marker_pushoff_peak(subject_id, trial_name, side, window):
    force_mot = DEFAULT.data_root / subject_id / "PROCESSED" / "MOCAP" / "force_data" / f"{trial_name}_forces_PROCESSED.mot"
    npz_path = DEFAULT.results_root / subject_id / "framework" / "single_leg_jump" / "noexo" / trial_name / "trial_data.npz"
    data = np.load(npz_path, allow_pickle=True)
    time, force = data["time"], data["force"]

    force_plate = read_storage(force_mot)
    plate = 1 if side == "R" else 2
    marker_vertical = np.abs(force_plate[f"ground_force_{plate}_vy"].to_numpy(float))
    shift = float(time[np.argmax(force)] - force_plate.time.iloc[np.argmax(marker_vertical)])

    marker_time = force_plate.time.to_numpy(float)
    window_marker = [window[0] - shift, window[1] - shift]
    mask = (marker_time >= window_marker[0]) & (marker_time <= window_marker[1])
    seg = marker_vertical[mask]
    return float(np.max(seg)) if seg.size else np.nan


def raw_insole_pushoff_peak(subject, trial, side, window):
    npz_path = trial.output / "trial_data.npz"
    data = np.load(npz_path, allow_pickle=True)
    shift_insole = float(data["shift_s"])
    insole_all = read_insole(trial.insole)
    insole_time = insole_all.time.to_numpy(float)
    column = "right_force" if side == "R" else "left_force"
    own_force = insole_all[column].to_numpy(float)
    window_insole = [window[0] - shift_insole, window[1] - shift_insole]
    mask = (insole_time >= window_insole[0]) & (insole_time <= window_insole[1])
    seg = own_force[mask]
    return float(np.max(seg)) if seg.size else np.nan


def main():
    plan = json.loads(PLAN_PATH.read_text())
    discovery = TrialDiscovery(DEFAULT)
    activity = activity_for("single_leg_jump", DEFAULT)

    rows = []
    for entry in plan:
        trial_name, side, subject_id = entry["trial_name"], entry["side"], entry["subject"]
        out_dir = RESULTS_DIR / f"{trial_name}_pushoff"
        grf_path = out_dir / "GRF.csv"
        status = "converged"
        if not grf_path.is_file():
            grf_path = out_dir / "GRF_NonConvergent.csv"
            status = "non_convergent"
        if not grf_path.is_file():
            print(f"{trial_name}: no result at all -- skipped")
            continue

        grf = pd.read_csv(grf_path)
        word = "right" if side == "R" else "left"
        opensimad_peak = float(np.max(np.abs(grf[f"ground_force_{word}_vy"].to_numpy(float))))

        marker_peak = marker_pushoff_peak(subject_id, trial_name, side, entry["time_window"])

        subject = Subject(subject_id, DEFAULT)
        trial = next(t for t in discovery.trials(subject, activity, "NOEXO", for_analysis=True) if t.name == trial_name)
        raw_peak = raw_insole_pushoff_peak(subject, trial, side, entry["time_window"])

        opensimad_gain = opensimad_peak / raw_peak
        marker_gain = marker_peak / raw_peak
        rows.append({
            "Trial": trial_name, "Side": side, "Status": status,
            "OpenSimAD_N": round(opensimad_peak, 1), "Marker_N": round(marker_peak, 1),
            "PctError": round(100 * (opensimad_peak - marker_peak) / marker_peak, 1),
            "RawInsole_N": round(raw_peak, 1),
            "OpenSimAD_gain": round(opensimad_gain, 3), "Marker_gain": round(marker_gain, 3),
            "GainPctDiff": round(100 * (opensimad_gain - marker_gain) / marker_gain, 1),
        })

    table = pd.DataFrame(rows)
    out_path = Path(__file__).resolve().parent / "pushoff_comparison.csv"
    table.to_csv(out_path, index=False)
    print(table.to_string(index=False))
    print(f"\nSaved -> {out_path}")

    converged = table[table.Status == "converged"]
    if len(converged):
        print(f"\nConverged only (n={len(converged)}): mean |PctError|={converged.PctError.abs().mean():.1f}%, "
              f"mean |GainPctDiff|={converged.GainPctDiff.abs().mean():.1f}%")
    print(f"All (n={len(table)}): mean |PctError|={table.PctError.abs().mean():.1f}%, "
          f"mean |GainPctDiff|={table.GainPctDiff.abs().mean():.1f}%")


if __name__ == "__main__":
    main()
