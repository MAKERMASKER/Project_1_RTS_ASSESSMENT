"""framework_rms_with_opensimad_gain.py -- for each trial with a known
OpenSimAD-derived push-off gain, temporarily swap that gain into the
subject's calibration file, rerun the REAL FrameworkProcessor pipeline
(full sync/shift/contact-masking, the same one production uses) for that
one trial, and read the already-established GrfRmse_N (vertical GRF RMSE
over the whole synced analysis window, vs. marker ground truth) via
OutcomeExtractor.analyze_trial() -- then compare against the SAME metric
computed with the current (stance-based) production gain. Always restores
the original calibration file afterward -- nothing here is left applied.

This answers "does substituting the OpenSimAD gain actually improve the
metric this project already trusts" directly, rather than an ad hoc
window-only trace comparison.

Run (from code/, in the opensimPy environment):
    python expt_temp/opensimad_gain_check/framework_rms_with_opensimad_gain.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from config import DEFAULT  # noqa: E402
from subject import Subject  # noqa: E402
from activities import activity_for  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from framework_processor import FrameworkProcessor  # noqa: E402
from outcomes import OutcomeExtractor, METRIC_COLUMNS  # noqa: E402

# (subject, side): OpenSimAD peak-based push-off gain -- the currently
# available set (AB05 still processing, excluded until it lands).
OPENSIMAD_GAINS = {
    ("AB03", "L"): 0.731,
    ("AB03", "R"): 0.874,
    ("AB07", "L"): 1.070,
    ("AB07", "R"): 1.053,
    ("AB08", "L"): 0.709,
    ("AB08", "R"): 0.782,
    ("AB10", "L"): 1.291,
    ("AB10", "R"): 1.215,
}

TRIAL_NAMES = {
    ("AB03", "L"): "AB03_NOEXO_SLJ_L_3",
    ("AB03", "R"): "AB03_NOEXO_SLJ_R_1",
    ("AB07", "L"): "AB07_NOEXO_SLJ_L_1",
    ("AB07", "R"): "AB07_NOEXO_SLJ_R_1",
    ("AB08", "L"): "AB08_NOEXO_SLJ_L_1",
    ("AB08", "R"): "AB08_NOEXO_SLJ_R_2",
    ("AB10", "L"): "AB10_NOEXO_SLJ_L_1",
    ("AB10", "R"): "AB10_NOEXO_SLJ_R_1",
}


def get_grf_rmse(subject_id, trial_name):
    activity = activity_for("single_leg_jump", DEFAULT)
    subject = Subject(subject_id, DEFAULT)
    discovery = TrialDiscovery(DEFAULT)
    trial = next(t for t in discovery.trials(subject, activity, "all", for_analysis=True) if t.name == trial_name)

    FrameworkProcessor(DEFAULT).process(subject, activity, trial)
    result = OutcomeExtractor(DEFAULT).analyze_trial(subject, activity, trial, make_plot=False)
    if result is None:
        return None
    row, _series = result
    return dict(zip(METRIC_COLUMNS, row))


def main():
    rows = []
    for (subject_id, side), opensimad_gain in OPENSIMAD_GAINS.items():
        trial_name = TRIAL_NAMES[(subject_id, side)]
        calibration_path = DEFAULT.calibration_root / f"{subject_id}_insole_marker_foot_calibration.json"
        original_text = calibration_path.read_text()

        print(f"=== {trial_name} (side={side}) ===")
        before = get_grf_rmse(subject_id, trial_name)
        calibration = json.loads(original_text)
        current_production_gain = calibration[side]["vertical_force_scale"]

        try:
            calibration[side] = dict(calibration[side])
            calibration[side]["vertical_force_scale"] = opensimad_gain
            calibration_path.write_text(json.dumps(calibration, indent=2))
            after = get_grf_rmse(subject_id, trial_name)
        finally:
            calibration_path.write_text(original_text)
            # Restore production trial_data.npz too -- re-run once more
            # with the original calibration back in place so nothing is
            # left in the OpenSimAD-gain state on disk.
            get_grf_rmse(subject_id, trial_name)

        if before is None or after is None:
            print(f"  skipped ({trial_name} not analyzable -- quality_omit or missing files)")
            continue

        rows.append({
            "Trial": trial_name, "Side": side,
            "ProductionGain": current_production_gain, "OpenSimADGain": opensimad_gain,
            "GrfRmse_N_production": round(before["GrfRmse_N"], 1),
            "GrfRmse_N_opensimad": round(after["GrfRmse_N"], 1),
            "PeakFrameworkGRF_N_production": round(before["PeakFrameworkGRF_N"], 1),
            "PeakFrameworkGRF_N_opensimad": round(after["PeakFrameworkGRF_N"], 1),
            "PeakMarkerGRF_N": round(before["PeakMarkerGRF_N"], 1),
            "Improved": after["GrfRmse_N"] < before["GrfRmse_N"],
        })
        print(f"  production gain={current_production_gain}: GrfRmse_N={before['GrfRmse_N']:.1f}")
        print(f"  OpenSimAD gain={opensimad_gain}: GrfRmse_N={after['GrfRmse_N']:.1f}")

    table = pd.DataFrame(rows)
    out_path = Path(__file__).resolve().parent / "framework_rms_with_opensimad_gain.csv"
    table.to_csv(out_path, index=False)
    print(f"\n{table.to_string(index=False)}")
    print(f"\nSaved -> {out_path}")
    if len(table):
        print(f"\nImproved: {table.Improved.sum()}/{len(table)}")


if __name__ == "__main__":
    main()
