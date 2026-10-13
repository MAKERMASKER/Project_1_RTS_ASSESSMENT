"""analyze_pilot.py -- compare OpenSimAD's kinematics-only GRF estimate
against (a) marker ground truth (from this project's own trial_summary.csv)
and (b) the raw insole reading, for one converged pilot trial. Computes the
OpenSimAD-implied gain the same way the "ideal" (marker-based) gain was
computed earlier this session: peak / raw_insole_peak.

Run (from code/, in the opensimPy environment):
    python expt_temp/opensimad_gain_check/analyze_pilot.py <trial_name> <side>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from config import DEFAULT  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from activities import activity_for  # noqa: E402
from subject import Subject  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main():
    trial_name, side = sys.argv[1], sys.argv[2]
    subject_id = trial_name.split("_")[0]

    grf_path = RESULTS_DIR / trial_name / "GRF.csv"
    grf = pd.read_csv(grf_path)
    word = "right" if side == "R" else "left"
    vertical = grf[f"ground_force_{word}_vy"].to_numpy(float)
    opensimad_peak = float(np.max(np.abs(vertical)))
    print(f"{trial_name} ({side}): OpenSimAD peak vertical GRF = {opensimad_peak:.1f} N")

    summary_path = DEFAULT.results_root / subject_id / "analysis" / "single_leg_jump" / "trial_summary.csv"
    marker_peak = None
    if summary_path.is_file():
        summary = pd.read_csv(summary_path)
        row = summary[summary.Trial == trial_name]
        if len(row):
            marker_peak = float(row.iloc[0]["PeakMarkerGRF_N"])
            print(f"  Marker ground truth peak = {marker_peak:.1f} N "
                  f"(OpenSimAD vs marker: {100 * (opensimad_peak - marker_peak) / marker_peak:+.1f}%)")

    rules = json.loads((CODE_ROOT / "subject_rules" / f"{subject_id}.json").read_text())
    fitted_gain = rules.get("force_scale", {}).get(side)
    if fitted_gain is None:
        calibration = json.loads((DEFAULT.calibration_root / f"{subject_id}_insole_marker_foot_calibration.json").read_text())
        fitted_gain = calibration[side]["vertical_force_scale"]

    package = DEFAULT.results_root / subject_id / "framework" / "single_leg_jump"
    condition_dir = "noexo" if "NOEXO" in trial_name else "exo"
    npz_path = package / condition_dir / trial_name / "trial_data.npz"
    data = np.load(npz_path, allow_pickle=True)
    framework_peak = float(np.max(np.abs(data["force"])))
    raw_insole_peak = framework_peak / fitted_gain
    print(f"  Raw insole peak (undoing current fitted gain {fitted_gain:.4f}) = {raw_insole_peak:.1f} N")

    opensimad_implied_gain = opensimad_peak / raw_insole_peak
    print(f"  OpenSimAD-implied gain = {opensimad_implied_gain:.3f}")
    if marker_peak is not None:
        marker_implied_gain = marker_peak / raw_insole_peak
        print(f"  Marker-implied ('ideal') gain = {marker_implied_gain:.3f}")
        print(f"  OpenSimAD-implied vs marker-implied gain difference: "
              f"{100 * (opensimad_implied_gain - marker_implied_gain) / marker_implied_gain:+.1f}%")


if __name__ == "__main__":
    main()
