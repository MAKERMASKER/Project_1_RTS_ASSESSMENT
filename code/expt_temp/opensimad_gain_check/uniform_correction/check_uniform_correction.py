"""check_uniform_correction.py -- test a single, uniform correction factor
applied to every subject's OpenSimAD push-off gain, rather than trusting
each subject's raw OpenSimAD-derived gain individually (which isn't a real
field-deployable rule -- you can't know in advance which subjects it will
help or hurt without ground truth).

CORRECTION_FACTOR = 1.156, derived from the mean gain gap (OpenSimAD LSQ
gain vs. marker LSQ gain, see analyze_pushoff_rms.py) across the 7
converged trials that did NOT need a ROM-cap rework (AB03_NOEXO_SLJ_L_3
excluded as a distinct failure mode -- its solve fought a joint-bound
issue, unlike every other trial): mean gap -13.5%, std 4.3%. Applying
`corrected_gain = raw_opensimad_gain / 0.865` should bring every subject
close to the truth WITHOUT subject-specific tuning, if this bias is
genuinely a property of OpenSimAD's own systematic behavior rather than
per-subject noise.

For each of the 8 trials with a known OpenSimAD gain, this reruns the REAL
production FrameworkProcessor + OutcomeExtractor (with plots) three ways:
  1. production (current subject_rules gain, unchanged)
  2. raw OpenSimAD gain (no correction -- already tested once before,
     rerun here just so all three plots live together for comparison)
  3. corrected OpenSimAD gain (raw * 1.156)
Always restores the original calibration file and regenerates the
production trial_data.npz/plot afterward -- nothing is left altered.

Run (from code/, in the opensimPy environment):
    python expt_temp/opensimad_gain_check/uniform_correction/check_uniform_correction.py
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_ROOT))

from config import DEFAULT  # noqa: E402
from subject import Subject  # noqa: E402
from activities import activity_for  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from framework_processor import FrameworkProcessor  # noqa: E402
from outcomes import OutcomeExtractor, METRIC_COLUMNS  # noqa: E402

CORRECTION_FACTOR = 1.156

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

OUT_DIR = Path(__file__).resolve().parent / "results"


def run_variant(subject_id, trial_name, make_plot):
    activity = activity_for("single_leg_jump", DEFAULT)
    subject = Subject(subject_id, DEFAULT)
    discovery = TrialDiscovery(DEFAULT)
    trial = next(t for t in discovery.trials(subject, activity, "all", for_analysis=True) if t.name == trial_name)

    FrameworkProcessor(DEFAULT).process(subject, activity, trial)
    result = OutcomeExtractor(DEFAULT).analyze_trial(subject, activity, trial, make_plot=make_plot)
    if result is None:
        return None
    row, _series = result
    return dict(zip(METRIC_COLUMNS, row))


def production_plot_path(subject_id, trial_name):
    return (
        DEFAULT.results_root / subject_id / "analysis" / "single_leg_jump" / "trial_comparisons"
        / f"{trial_name}_sync_grf_cop_angles_id.png"
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for (subject_id, side), raw_gain in OPENSIMAD_GAINS.items():
        trial_name = TRIAL_NAMES[(subject_id, side)]
        corrected_gain = round(raw_gain * CORRECTION_FACTOR, 4)
        calibration_path = DEFAULT.calibration_root / f"{subject_id}_insole_marker_foot_calibration.json"
        original_text = calibration_path.read_text()
        calibration = json.loads(original_text)
        production_gain = calibration[side]["vertical_force_scale"]

        print(f"=== {trial_name} (side={side}) -- raw={raw_gain}, corrected={corrected_gain} ===")
        trial_out = OUT_DIR / trial_name
        trial_out.mkdir(parents=True, exist_ok=True)

        try:
            # 1. production baseline
            production = run_variant(subject_id, trial_name, make_plot=True)
            shutil.copy2(production_plot_path(subject_id, trial_name), trial_out / f"{trial_name}_1_production.png")

            # 2. raw OpenSimAD gain
            calibration[side] = dict(calibration[side])
            calibration[side]["vertical_force_scale"] = raw_gain
            calibration_path.write_text(json.dumps(calibration, indent=2))
            raw_result = run_variant(subject_id, trial_name, make_plot=True)
            shutil.copy2(production_plot_path(subject_id, trial_name), trial_out / f"{trial_name}_2_raw_opensimad.png")

            # 3. corrected (uniform factor) OpenSimAD gain
            calibration[side]["vertical_force_scale"] = corrected_gain
            calibration_path.write_text(json.dumps(calibration, indent=2))
            corrected_result = run_variant(subject_id, trial_name, make_plot=True)
            shutil.copy2(production_plot_path(subject_id, trial_name), trial_out / f"{trial_name}_3_corrected_opensimad.png")
        finally:
            calibration_path.write_text(original_text)
            run_variant(subject_id, trial_name, make_plot=True)  # regenerate the real production state

        if production is None or raw_result is None or corrected_result is None:
            print(f"  skipped ({trial_name} not analyzable for one variant)")
            continue

        rows.append({
            "Trial": trial_name, "Side": side,
            "ProductionGain": round(production_gain, 4),
            "RawOpenSimADGain": raw_gain, "CorrectedOpenSimADGain": corrected_gain,
            "GrfRmse_N_production": round(production["GrfRmse_N"], 1),
            "GrfRmse_N_raw_opensimad": round(raw_result["GrfRmse_N"], 1),
            "GrfRmse_N_corrected_opensimad": round(corrected_result["GrfRmse_N"], 1),
            "PeakMarkerGRF_N": round(production["PeakMarkerGRF_N"], 1),
            "PeakFrameworkGRF_N_production": round(production["PeakFrameworkGRF_N"], 1),
            "PeakFrameworkGRF_N_corrected": round(corrected_result["PeakFrameworkGRF_N"], 1),
            "CorrectedBetterThanProduction": corrected_result["GrfRmse_N"] < production["GrfRmse_N"],
            "CorrectedBetterThanRaw": corrected_result["GrfRmse_N"] < raw_result["GrfRmse_N"],
        })
        print(f"  production: {production['GrfRmse_N']:.1f}  raw_opensimad: {raw_result['GrfRmse_N']:.1f}  "
              f"corrected_opensimad: {corrected_result['GrfRmse_N']:.1f}")

    table = pd.DataFrame(rows)
    out_path = OUT_DIR / "uniform_correction_summary.csv"
    table.to_csv(out_path, index=False)
    print(f"\n{table.to_string(index=False)}")
    print(f"\nSaved -> {out_path}")
    if len(table):
        print(f"\nCorrected better than production: {table.CorrectedBetterThanProduction.sum()}/{len(table)}")
        print(f"Corrected better than raw OpenSimAD: {table.CorrectedBetterThanRaw.sum()}/{len(table)}")


if __name__ == "__main__":
    main()
