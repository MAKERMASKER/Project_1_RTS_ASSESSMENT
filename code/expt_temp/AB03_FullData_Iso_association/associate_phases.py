"""associate_phases.py -- AB03: phase/mode-separated SLJ LSI against isokinetic
extension AND flexion, with specificity measured rather than assumed.

Every SLJ metric is compared against BOTH isokinetic groups, not only the one
mechanism predicts. That turns "does SLJ extension track isokinetic extension"
into a testable claim: if a metric sits equally close to extension and flexion,
the association is not muscle-specific and is more likely subject-level
scaling (a bigger, stronger person scores higher on everything). SPECIFICITY
below is the signed margin `|SLJ - wrong group| - |SLJ - predicted group|`;
positive means the mechanism-predicted pairing really is closer.

Mode matching, stated per metric rather than averaged over:
  * PUSH-OFF is CONCENTRIC and the isokinetic reference is `Con/Con`, so these
    pairings are mode-matched.
  * LANDING is ECCENTRIC. Nothing in a Con/Con protocol matches it. Those rows
    are reported with MatchedMode=False; a large disagreement there is a
    finding about eccentric-vs-concentric capacity, not a measurement failure,
    and is exactly what an isokinetic screen cannot see.

Bootstrap 95% CIs over the 5 trials per group, because at n=1 subject a point
LSI carries no usable uncertainty on its own.

Run from code/:
    ..\\opensimPy\\Scripts\\python.exe expt_temp/AB03_FullData_Iso_association/associate_phases.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(CODE_ROOT))

from config import DEFAULT  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
WORKBOOK = DEFAULT.data_root / "Isokinetic_Study_AB02_to_AB09_VALIDATED.xlsx"
SUBJECT_ID = "AB03"
BOOTSTRAP = 5000
RANDOM_SEED = 20260930
X_LIMIT = 200.0

COMPARISONS = [
    ("NOEXO_L_vs_R", "NOEXO", "L", "NOEXO", "R"),
    ("NOEXO_L_vs_EXO_R", "NOEXO", "L", "EXO", "R"),
    ("EXO_R_vs_NOEXO_R", "EXO", "R", "NOEXO", "R"),
]

# (movement stem, label, phase, mode-matched?, predicted isokinetic group, isokinetic kind)
METRICS = [
    ("PushOff_KneeConcWork_J", "Push-off knee concentric work", "PushOff", True, "Extension", "TotalWork_ftlb"),
    ("PushOff_KneeExtPeak_Nm", "Push-off knee extensor peak", "PushOff", True, "Extension", "PeakTorque_ftlb"),
    ("PushOff_AnkleConcWork_J", "Push-off ankle concentric work", "PushOff", True, "Extension", "TotalWork_ftlb"),
    ("PushOff_SupportPeak_Nm", "Push-off support moment peak", "PushOff", True, "Extension", "PeakTorque_ftlb"),
    ("PushOff_HipConcWork_J", "Push-off hip concentric work", "PushOff", True, "Flexion", "TotalWork_ftlb"),
    ("PushOff_HipExtPeak_Nm", "Push-off hip extensor peak", "PushOff", True, "Flexion", "PeakTorque_ftlb"),
    ("Landing_KneeEccWork_J", "Landing knee eccentric work", "Landing", False, "Extension", "TotalWork_ftlb"),
    ("Landing_KneeExtPeak_Nm", "Landing knee extensor peak", "Landing", False, "Extension", "PeakTorque_ftlb"),
    ("Landing_SupportPeak_Nm", "Landing support moment peak", "Landing", False, "Extension", "PeakTorque_ftlb"),
    ("Landing_HipEccWork_J", "Landing hip eccentric work", "Landing", False, "Flexion", "TotalWork_ftlb"),
]


def _lsi_with_ci(test_values, reference_values, rng):
    test_values = np.asarray(test_values, float)
    reference_values = np.asarray(reference_values, float)
    test_values = test_values[np.isfinite(test_values)]
    reference_values = reference_values[np.isfinite(reference_values)]
    if not len(test_values) or not len(reference_values) or reference_values.mean() == 0:
        return np.nan, np.nan, np.nan
    point = 100 * test_values.mean() / reference_values.mean()
    draws = np.empty(BOOTSTRAP)
    for i in range(BOOTSTRAP):
        a = rng.choice(test_values, len(test_values), replace=True).mean()
        b = rng.choice(reference_values, len(reference_values), replace=True).mean()
        draws[i] = 100 * a / b if b else np.nan
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return point, low, high


def main():
    kinetics = pd.read_csv(RESULTS / "ab03_phase_kinetics.csv")
    iso = pd.read_excel(WORKBOOK, "Isokinetic_LSI")
    iso = iso[iso.Subject == SUBJECT_ID]
    rng = np.random.default_rng(RANDOM_SEED)

    def iso_value(comparison, group, kind):
        hit = iso[(iso.Comparison == comparison) & (iso.Metric == f"{group}_{kind}")]
        return float(hit.LSI_pct.iloc[0]) if len(hit) else np.nan

    rows = []
    for comparison, test_condition, test_side, reference_condition, reference_side in COMPARISONS:
        test = kinetics[(kinetics.Condition == test_condition) & (kinetics.Side == test_side)]
        reference = kinetics[(kinetics.Condition == reference_condition)
                             & (kinetics.Side == reference_side)]
        if not len(test) or not len(reference):
            continue
        for stem, label, phase, matched_mode, predicted, kind in METRICS:
            extension = iso_value(comparison, "Extension", kind)
            flexion = iso_value(comparison, "Flexion", kind)
            record = {
                "Comparison": comparison, "Phase": phase, "Metric": label,
                "Mode": "concentric" if phase == "PushOff" else "eccentric",
                "MatchedMode": matched_mode, "Predicted": predicted,
                "Iso_Extension_pct": round(extension, 1) if np.isfinite(extension) else np.nan,
                "Iso_Flexion_pct": round(flexion, 1) if np.isfinite(flexion) else np.nan,
            }
            for pipeline in ("Framework", "Marker"):
                column = f"{pipeline}_{stem}"
                if column not in kinetics.columns:
                    continue
                point, low, high = _lsi_with_ci(test[column], reference[column], rng)
                record[f"{pipeline}_LSI_pct"] = round(point, 1) if np.isfinite(point) else np.nan
                record[f"{pipeline}_CI_low"] = round(low, 1) if np.isfinite(low) else np.nan
                record[f"{pipeline}_CI_high"] = round(high, 1) if np.isfinite(high) else np.nan
                record[f"{pipeline}_vs_Ext"] = round(point - extension, 1) if np.isfinite(point) else np.nan
                record[f"{pipeline}_vs_Flex"] = round(point - flexion, 1) if np.isfinite(point) else np.nan
                if np.isfinite(point) and np.isfinite(extension) and np.isfinite(flexion):
                    to_predicted = abs(point - (extension if predicted == "Extension" else flexion))
                    to_other = abs(point - (flexion if predicted == "Extension" else extension))
                    record[f"{pipeline}_Specificity"] = round(to_other - to_predicted, 1)
            rows.append(record)

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "ab03_phase_association.csv", index=False)
    pd.set_option("display.width", 250)

    for comparison, *_ in COMPARISONS:
        part = table[table.Comparison == comparison]
        if not len(part):
            continue
        print(f"\n=== {comparison} ===")
        print(part[["Phase", "Metric", "Predicted", "Framework_LSI_pct", "Framework_CI_low",
                    "Framework_CI_high", "Marker_LSI_pct", "Iso_Extension_pct", "Iso_Flexion_pct",
                    "Framework_Specificity"]].to_string(index=False))

    print("\n=== mode matching: mean |SLJ - predicted isokinetic| ===")
    for pipeline in ("Framework", "Marker"):
        column = f"{pipeline}_vs_Ext"
        predicted_gap = table.apply(
            lambda r, p=pipeline: abs(r[f"{p}_vs_Ext"]) if r.Predicted == "Extension"
            else abs(r[f"{p}_vs_Flex"]), axis=1)
        grouped = table.assign(gap=predicted_gap).groupby(["MatchedMode"]).gap.mean().round(1)
        print(f"  {pipeline}: " + ", ".join(
            f"{'mode-matched (push-off)' if k else 'mode-MISmatched (landing)'} {v}"
            for k, v in grouped.items()))

    print("\n=== specificity: is the mechanism-predicted isokinetic group actually closer? ===")
    for pipeline in ("Framework", "Marker"):
        column = f"{pipeline}_Specificity"
        if column not in table:
            continue
        valid = table[np.isfinite(table[column])]
        wins = int((valid[column] > 0).sum())
        print(f"  {pipeline}: predicted group closer in {wins}/{len(valid)} rows "
              f"(mean margin {valid[column].mean():+.1f} LSI points)")
        by_group = valid.groupby("Predicted")[column].agg(["mean", "count"]).round(1)
        print(by_group.to_string())

    _plot(table)
    print(f"\n-> {RESULTS / 'ab03_phase_association.csv'}")


def _plot(table):
    out = RESULTS / "plots"
    out.mkdir(parents=True, exist_ok=True)
    comparisons = list(table.Comparison.unique())
    figure, axes = plt.subplots(len(comparisons), 1, figsize=(13, 4.8 * len(comparisons)),
                                squeeze=False)
    for index, comparison in enumerate(comparisons):
        axis = axes[index][0]
        part = table[table.Comparison == comparison].reset_index(drop=True)
        y = np.arange(len(part))
        axis.errorbar(part.Framework_LSI_pct, y - 0.16,
                      xerr=[part.Framework_LSI_pct - part.Framework_CI_low,
                            part.Framework_CI_high - part.Framework_LSI_pct],
                      fmt="o", color="red", capsize=3, markersize=5, label="Framework (95% CI)")
        axis.errorbar(part.Marker_LSI_pct, y,
                      xerr=[part.Marker_LSI_pct - part.Marker_CI_low,
                            part.Marker_CI_high - part.Marker_LSI_pct],
                      fmt="s", color="black", capsize=3, markersize=5, label="Marker (95% CI)")
        axis.scatter(part.Iso_Extension_pct, y + 0.16, marker="D", color="darkorange", s=42,
                     zorder=5, label="Isokinetic EXTENSION")
        axis.scatter(part.Iso_Flexion_pct, y + 0.30, marker="v", color="seagreen", s=42,
                     zorder=5, label="Isokinetic FLEXION")
        axis.axvline(100, color="grey", ls="--", lw=1)
        axis.set_yticks(y)
        axis.set_yticklabels(
            [f"[{p[:4]}|{'con' if p == 'PushOff' else 'ecc'}] {m}  (->{g})"
             for p, m, g in zip(part.Phase, part.Metric, part.Predicted)], fontsize=8)
        axis.invert_yaxis()
        axis.set_xlim(0, X_LIMIT)
        axis.set_xlabel("LSI (%)")
        axis.set_title(f"AB03 {comparison}   (-> marks the mechanism-predicted isokinetic group)")
        axis.grid(True, alpha=0.3, axis="x")
        axis.legend(fontsize=7, loc="lower right")
    figure.tight_layout()
    figure.savefig(out / "ab03_phase_association.png", dpi=140)
    plt.close(figure)
    print(f"-> {out / 'ab03_phase_association.png'}")


if __name__ == "__main__":
    main()
