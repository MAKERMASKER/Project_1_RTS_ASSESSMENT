"""sweep_stance_lowpass.py -- re-run InsoleCalibration's AUTOMATIC quality-
gated stance-window detector (calibration.py's _stance_window) at a higher
low-pass cutoff (12 Hz vs. the current 6 Hz), across every lab subject/side,
using the SAME auto-fit formula (gain = body_weight / median_of_accepted)
-- bypassing subject_rules' "fixed" manual overrides so this is a clean,
like-for-like comparison of the automatic detector at two cutoffs, not
mixed with any manually-selected windows.

Purpose: check whether a higher cutoff changes the resulting per-subject/
side gain enough to correlate with body weight (mass_kg) or shoe size --
the user's stated goal being a gain identifiable from body weight alone
(for eventual field use without marker ground truth), not just a per-
subject fit. Also reports each cutoff's gain against the same "ideal"
landing-peak-implied gain computed earlier (real SLJ trials' marker peak
/ raw insole peak) for a direct accuracy comparison, not just a body-
weight-correlation check.

Run (from code/, in the opensimPy environment):
    python expt_temp/insole_gain_body_weight/sweep_stance_lowpass.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from calibration import InsoleCalibration  # noqa: E402
from config import DEFAULT  # noqa: E402
from subject import Subject  # noqa: E402

SUBJECT_IDS = ("AB03", "AB05", "AB07", "AB08", "AB10")
CUTOFFS_HZ = (6.0, 12.0)

# Ideal (landing-peak-implied) gains already derived from real SLJ trials
# (marker peak / raw insole peak, undoing whatever gain trial_data.npz was
# built with) -- see this session's own investigation. Kept here as a
# fixed reference table, not recomputed by this script.
IDEAL_GAIN = {
    ("AB03", "L"): 0.830, ("AB03", "R"): 0.822,
    ("AB05", "L"): 0.696, ("AB05", "R"): 0.819,
    ("AB07", "L"): 1.019, ("AB07", "R"): 1.100,
    ("AB08", "L"): 0.645, ("AB08", "R"): 0.745,
    ("AB10", "L"): 1.035, ("AB10", "R"): 1.051,
}


class _SweepCalibration(InsoleCalibration):
    pass


def main():
    meta = pd.read_csv(DEFAULT.data_root / "subject_metadata.csv").set_index("SubjectID")
    rows = []
    for cutoff in CUTOFFS_HZ:
        calibrator = _SweepCalibration(DEFAULT)
        calibrator.STANCE_LOWPASS_HZ = cutoff
        for subject_id in SUBJECT_IDS:
            subject = Subject(subject_id, DEFAULT)
            rules = dict(subject.rules)
            rules["force_scale_mode"] = {"L": InsoleCalibration.AUTO_MODE, "R": InsoleCalibration.AUTO_MODE}
            for side in ("L", "R"):
                try:
                    gain, source, weight, median, count, _reports = calibrator._force_gain(subject, side, rules)
                except ValueError as error:
                    rows.append({
                        "CutoffHz": cutoff, "Subject": subject_id, "Side": side,
                        "Mass_kg": meta.loc[subject_id, "Mass_kg"], "ShoeSize": meta.loc[subject_id, "ShoeSize"],
                        "Gain": np.nan, "AcceptedTrials": 0, "Error": str(error),
                    })
                    continue
                rows.append({
                    "CutoffHz": cutoff, "Subject": subject_id, "Side": side,
                    "Mass_kg": meta.loc[subject_id, "Mass_kg"], "ShoeSize": meta.loc[subject_id, "ShoeSize"],
                    "Gain": round(gain, 3), "AcceptedTrials": count,
                    "IdealGain": IDEAL_GAIN.get((subject_id, side), np.nan),
                    "Error": None,
                })

    table = pd.DataFrame(rows)
    out_path = Path(__file__).resolve().parent / "stance_lowpass_sweep.csv"
    table.to_csv(out_path, index=False)
    print(table.to_string(index=False))
    print(f"\nSaved -> {out_path}\n")

    for cutoff in CUTOFFS_HZ:
        sub = table[(table.CutoffHz == cutoff) & table.Gain.notna()]
        if len(sub) < 2:
            print(f"{cutoff} Hz: not enough valid rows to correlate")
            continue
        mass_corr = sub["Mass_kg"].corr(sub["Gain"])
        shoe_corr = sub["ShoeSize"].corr(sub["Gain"])
        ideal_diff = (sub["Gain"] - sub["IdealGain"]).abs().mean() if "IdealGain" in sub else np.nan
        print(f"{cutoff} Hz: corr(mass, gain)={mass_corr:.3f}  corr(shoe, gain)={shoe_corr:.3f}  "
              f"mean|gain-ideal|={ideal_diff:.3f}  n={len(sub)}")


if __name__ == "__main__":
    main()
