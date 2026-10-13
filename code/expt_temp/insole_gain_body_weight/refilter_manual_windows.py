"""refilter_manual_windows.py -- recompute each subject's ALREADY-SELECTED
manual stance windows (Insole_vertical_force_calib_window/select_stance_
windows.py's own *_manual_stance_selections.csv, human-verified genuine
single-leg stance spans) at a HIGHER low-pass cutoff (12 Hz vs. the
select_stance_windows.py tool's own 6 Hz), to see whether that changes the
resulting per-subject/side gain enough to correlate with body weight or
shoe size.

Why re-filter instead of re-running the automatic quality-gated detector
at 12 Hz: that detector (calibration.py's _stance_window) barely finds any
candidate window at all on this project's SLJ data regardless of cutoff
(confirmed: identical results at 6 Hz and 12 Hz for every subject except
2 of 10 sides) -- the bottleneck there is the duration/CV/weight-fraction
GATING finding nothing, not the filter frequency. The human-selected
windows are real, confirmed-good stance spans, so re-filtering THEM in
isolation is the only way to actually test whether the cutoff itself
matters, isolated from the gating question.

Run (from code/, in the opensimPy environment):
    python expt_temp/insole_gain_body_weight/refilter_manual_windows.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from config import DEFAULT  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from activities import activity_for  # noqa: E402
from io_tables import read_insole  # noqa: E402
from opensim_env import lowpass  # noqa: E402
from subject import Subject  # noqa: E402

SELECTIONS_DIR = CODE_ROOT / "expt_temp" / "Insole_vertical_force_calib_window"
OUT_DIR = Path(__file__).resolve().parent
SUBJECT_IDS = ("AB03", "AB05", "AB07", "AB08", "AB10")
CUTOFFS_HZ = (6.0, 12.0)

IDEAL_GAIN = {
    ("AB03", "L"): 0.830, ("AB03", "R"): 0.822,
    ("AB05", "L"): 0.696, ("AB05", "R"): 0.819,
    ("AB07", "L"): 1.019, ("AB07", "R"): 1.100,
    ("AB08", "L"): 0.645, ("AB08", "R"): 0.745,
    ("AB10", "L"): 1.035, ("AB10", "R"): 1.051,
}


def _trial_lookup(subject_id):
    subject = Subject(subject_id, DEFAULT)
    activity = activity_for("single_leg_jump", DEFAULT)
    discovery = TrialDiscovery(DEFAULT)
    return subject, {trial.name: trial for trial in discovery.trials(subject, activity, "all", for_analysis=True)}


def main():
    meta = pd.read_csv(DEFAULT.data_root / "subject_metadata.csv").set_index("SubjectID")
    rows = []
    for subject_id in SUBJECT_IDS:
        selections_path = SELECTIONS_DIR / f"{subject_id}_manual_stance_selections.csv"
        if not selections_path.is_file():
            print(f"{subject_id}: no manual selections file -- skipped")
            continue
        selections = pd.read_csv(selections_path)
        accepted = selections[selections["Selected"] == True]  # noqa: E712
        if not len(accepted):
            continue

        subject, trials_by_name = _trial_lookup(subject_id)
        weight_n = subject.mass_kg * DEFAULT.gravity_mps2

        for side in ("L", "R"):
            side_rows = accepted[accepted["Side"] == side]
            if not len(side_rows):
                continue
            per_cutoff_medians = {cutoff: [] for cutoff in CUTOFFS_HZ}
            for _, row in side_rows.iterrows():
                trial = trials_by_name.get(row["Trial"])
                if trial is None:
                    continue
                insole = read_insole(trial.insole)
                time = insole["time"].to_numpy(float)
                column = "left_force" if side == "L" else "right_force"
                own = insole[column].to_numpy(float)
                mask = (time >= row["StartS"]) & (time <= row["EndS"])
                for cutoff in CUTOFFS_HZ:
                    filtered = lowpass(own, time, cutoff)
                    per_cutoff_medians[cutoff].append(float(np.median(filtered[mask])))

            for cutoff in CUTOFFS_HZ:
                medians = per_cutoff_medians[cutoff]
                if not medians:
                    continue
                aggregate_median = float(np.median(medians))
                gain = weight_n / aggregate_median
                rows.append({
                    "CutoffHz": cutoff, "Subject": subject_id, "Side": side,
                    "Mass_kg": meta.loc[subject_id, "Mass_kg"], "ShoeSize": meta.loc[subject_id, "ShoeSize"],
                    "Gain": round(gain, 4), "AcceptedTrials": len(medians),
                    "IdealGain": IDEAL_GAIN.get((subject_id, side), np.nan),
                })

    table = pd.DataFrame(rows)
    out_path = OUT_DIR / "manual_window_refilter_sweep.csv"
    table.to_csv(out_path, index=False)
    print(table.to_string(index=False))
    print(f"\nSaved -> {out_path}\n")

    for cutoff in CUTOFFS_HZ:
        sub = table[table.CutoffHz == cutoff]
        if len(sub) < 2:
            continue
        mass_corr = sub["Mass_kg"].corr(sub["Gain"])
        shoe_corr = sub["ShoeSize"].corr(sub["Gain"])
        ideal_diff = (sub["Gain"] - sub["IdealGain"]).abs().mean()
        print(f"{cutoff} Hz: corr(mass, gain)={mass_corr:.3f}  corr(shoe, gain)={shoe_corr:.3f}  "
              f"mean|gain-ideal|={ideal_diff:.3f}  n={len(sub)}")

    # Per-subject gain shift between cutoffs, to see if 12 Hz moves gains
    # meaningfully at all (independent of the correlation question).
    pivot = table.pivot_table(index=["Subject", "Side"], columns="CutoffHz", values="Gain")
    pivot["Delta"] = pivot[CUTOFFS_HZ[1]] - pivot[CUTOFFS_HZ[0]]
    print("\nPer-subject/side gain shift (12Hz - 6Hz):")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
