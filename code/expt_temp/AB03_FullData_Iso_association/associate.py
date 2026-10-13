"""associate.py -- AB03: SLJ movement LSI (framework AND marker) against
isokinetic LSI, for EXTENSION, FLEXION and their H:Q ratio.

What this can and cannot answer
-------------------------------
This is ONE subject, so nothing here is a correlation. The question is
agreement: does AB03's SLJ-derived limb symmetry land on the same number as
its isokinetic limb symmetry, and does it do so for flexion as well as
extension?

Because a single LSI is a ratio of two 5-trial means, the interesting quantity
is not the point estimate but its uncertainty. Every movement LSI here carries
a bootstrap 95% CI resampled over trials. An LSI whose CI spans 100% cannot
distinguish asymmetry from noise, no matter how far its point estimate sits
from 100 -- and for the noisier metrics that is exactly what happens.

The isokinetic side reports its own CoV per metric, which is carried through
so a disagreement can be attributed rather than assumed to be the movement
pipeline's fault.

Comparisons follow config.lsi_definitions plus EXO_R_vs_NOEXO_R, the
within-limb device effect, which the isokinetic workbook supplies and the
movement pipeline does not yet define. Within-limb removes between-limb
anatomical variance and is the better-posed contrast.

Metric pairings, with their honesty caveats:
  knee extensor   <-> Extension_*   direct; SLJ landing is extensor-dominated
  knee flexor     <-> Flexion_*     direct but WEAK -- see README; the knee
                                    flexor moment in a landing is small and
                                    intermittent
  hip extensor    <-> Flexion_*     functional; hamstrings are biarticular hip
                                    extensors and that is their main jump role
  H:Q (movement)  <-> *_HQ_Ratio_*  inherits the knee-flexor weakness

Run from code/:
    ..\\opensimPy\\Scripts\\python.exe expt_temp/AB03_FullData_Iso_association/associate.py
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
X_LIMIT = 220.0  # LSI axis clip for readability; see _plot()
RANDOM_SEED = 20260930

# (name, test condition, test side, reference condition, reference side)
COMPARISONS = [
    ("NOEXO_L_vs_R", "NOEXO", "L", "NOEXO", "R"),
    ("NOEXO_L_vs_EXO_R", "NOEXO", "L", "EXO", "R"),
    ("EXO_R_vs_NOEXO_R", "EXO", "R", "NOEXO", "R"),
]

# movement column stem -> (label, isokinetic metric, muscle group, mapping note)
PAIRS = [
    ("KneeExtPeak_Nm", "Knee extensor peak", "Extension_PeakTorque_ftlb", "Extension", "direct"),
    ("KneeExtWork_J", "Knee extensor work", "Extension_TotalWork_ftlb", "Extension", "direct"),
    ("KneeExtImpulse_Nms", "Knee extensor impulse", "Extension_TotalWork_ftlb", "Extension", "direct-ish"),
    ("KneeFlexPeak_Nm", "Knee flexor peak", "Flexion_PeakTorque_ftlb", "Flexion", "direct but weak"),
    ("KneeFlexWork_J", "Knee flexor work", "Flexion_TotalWork_ftlb", "Flexion", "direct but weak"),
    ("HipExtPeak_Nm", "Hip extensor peak", "Flexion_PeakTorque_ftlb", "Flexion", "functional"),
    ("HipExtWork_J", "Hip extensor work", "Flexion_TotalWork_ftlb", "Flexion", "functional"),
    ("HQ_PeakRatio_pct", "H:Q peak ratio", "PeakTorque_HQ_Ratio_pct", "H:Q", "inherits flexor weakness"),
    ("HQ_WorkRatio_pct", "H:Q work ratio", "TotalWork_HQ_Ratio_pct", "H:Q", "inherits flexor weakness"),
    ("HipExtOverKneeExt_pct", "Hip-ext / knee-ext", "PeakTorque_HQ_Ratio_pct", "H:Q", "functional H:Q"),
]

ISO_COV = {  # isokinetic metric -> the raw CoV columns backing it
    "Extension_PeakTorque_ftlb": "Quad_PeakTorque_CoV",
    "Extension_TotalWork_ftlb": "Quad_TotalWork_CoV",
    "Flexion_PeakTorque_ftlb": "Ham_PeakTorque_CoV",
    "Flexion_TotalWork_ftlb": "Ham_TotalWork_CoV",
}


def _lsi_with_ci(test_values, reference_values, rng):
    """100 * mean(test)/mean(reference), with a bootstrap 95% CI over trials."""
    test_values = np.asarray(test_values, float)
    reference_values = np.asarray(reference_values, float)
    test_values = test_values[np.isfinite(test_values)]
    reference_values = reference_values[np.isfinite(reference_values)]
    if not len(test_values) or not len(reference_values) or reference_values.mean() == 0:
        return np.nan, np.nan, np.nan, np.nan
    point = 100 * test_values.mean() / reference_values.mean()
    draws = np.empty(BOOTSTRAP)
    for i in range(BOOTSTRAP):
        a = rng.choice(test_values, len(test_values), replace=True)
        b = rng.choice(reference_values, len(reference_values), replace=True)
        draws[i] = 100 * a.mean() / b.mean() if b.mean() else np.nan
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    # Trial-level spread of the ratio's numerator, as a stability read.
    spread = 100 * test_values.std(ddof=1) / test_values.mean() if test_values.mean() else np.nan
    return point, low, high, spread


def main():
    kinetics = pd.read_csv(RESULTS / "ab03_trial_kinetics.csv")
    iso_lsi = pd.read_excel(WORKBOOK, "Isokinetic_LSI")
    iso_lsi = iso_lsi[iso_lsi.Subject == SUBJECT_ID]
    iso_raw = pd.read_excel(WORKBOOK, "Raw_Isokinetic")
    iso_raw = iso_raw[iso_raw.Subject == SUBJECT_ID]
    rng = np.random.default_rng(RANDOM_SEED)

    rows = []
    for comparison, test_condition, test_side, reference_condition, reference_side in COMPARISONS:
        test = kinetics[(kinetics.Condition == test_condition) & (kinetics.Side == test_side)]
        reference = kinetics[(kinetics.Condition == reference_condition)
                             & (kinetics.Side == reference_side)]
        if not len(test) or not len(reference):
            continue
        for stem, label, iso_metric, group, note in PAIRS:
            record = {
                "Comparison": comparison, "Metric": label, "Group": group,
                "Mapping": note, "IsokineticMetric": iso_metric,
                "TestTrials": len(test), "ReferenceTrials": len(reference),
            }
            for pipeline in ("Framework", "Marker"):
                column = f"{pipeline}{stem}"
                if column not in kinetics.columns:
                    continue
                point, low, high, spread = _lsi_with_ci(test[column], reference[column], rng)
                record[f"{pipeline}_LSI_pct"] = round(point, 1) if np.isfinite(point) else np.nan
                record[f"{pipeline}_CI_low"] = round(low, 1) if np.isfinite(low) else np.nan
                record[f"{pipeline}_CI_high"] = round(high, 1) if np.isfinite(high) else np.nan
                record[f"{pipeline}_TrialCV_pct"] = round(spread, 1) if np.isfinite(spread) else np.nan
            hit = iso_lsi[(iso_lsi.Comparison == comparison) & (iso_lsi.Metric == iso_metric)]
            record["Isokinetic_LSI_pct"] = round(float(hit.LSI_pct.iloc[0]), 1) if len(hit) else np.nan

            cov_column = ISO_COV.get(iso_metric)
            if cov_column and cov_column in iso_raw.columns:
                both = iso_raw[
                    ((iso_raw.Condition == test_condition) & (iso_raw.Limb == test_side))
                    | ((iso_raw.Condition == reference_condition) & (iso_raw.Limb == reference_side))
                ][cov_column]
                record["Isokinetic_CoV_max"] = round(float(both.max()), 3) if len(both) else np.nan
            rows.append(record)

    table = pd.DataFrame(rows)
    for pipeline in ("Framework", "Marker"):
        table[f"{pipeline}_minus_Iso"] = (table[f"{pipeline}_LSI_pct"]
                                          - table["Isokinetic_LSI_pct"]).round(1)
        # A CI spanning 100 cannot distinguish asymmetry from noise.
        table[f"{pipeline}_CIspans100"] = (
            (table[f"{pipeline}_CI_low"] <= 100) & (table[f"{pipeline}_CI_high"] >= 100)
        )
    table.to_csv(RESULTS / "ab03_lsi_association.csv", index=False)

    pd.set_option("display.width", 260)
    for comparison, *_rest in COMPARISONS:
        part = table[table.Comparison == comparison]
        if not len(part):
            continue
        print(f"\n=== {comparison} ===")
        print(part[["Group", "Metric", "Framework_LSI_pct", "Framework_CI_low", "Framework_CI_high",
                    "Marker_LSI_pct", "Isokinetic_LSI_pct", "Framework_minus_Iso",
                    "Marker_minus_Iso", "Framework_CIspans100"]].to_string(index=False))

    print("\n=== how close is each pipeline to isokinetic, by muscle group ===")
    summary = table.groupby("Group")[["Framework_minus_Iso", "Marker_minus_Iso"]].agg(
        lambda v: round(float(np.nanmean(np.abs(v))), 1))
    summary.columns = ["Framework mean |diff|", "Marker mean |diff|"]
    print(summary.to_string())

    print("\n=== framework vs marker agreement (pipeline validity, isokinetic not involved) ===")
    delta = (table.Framework_LSI_pct - table.Marker_LSI_pct).abs()
    print(f"  mean |framework - marker| across all metrics: {np.nanmean(delta):.1f} LSI points")
    by_group = table.assign(d=delta).groupby("Group").d.mean().round(1)
    print(by_group.to_string())

    _plot(table)
    print(f"\n-> {RESULTS / 'ab03_lsi_association.csv'}")


def _plot(table):
    out = RESULTS / "plots"
    out.mkdir(parents=True, exist_ok=True)
    comparisons = [c for c in table.Comparison.unique()]
    figure, axes = plt.subplots(len(comparisons), 1, figsize=(13, 4.6 * len(comparisons)),
                                squeeze=False)
    for index, comparison in enumerate(comparisons):
        axis = axes[index][0]
        part = table[table.Comparison == comparison].reset_index(drop=True)
        y = np.arange(len(part))
        axis.errorbar(part.Framework_LSI_pct, y - 0.18,
                      xerr=[part.Framework_LSI_pct - part.Framework_CI_low,
                            part.Framework_CI_high - part.Framework_LSI_pct],
                      fmt="o", color="red", capsize=3, label="Framework (95% CI)", markersize=5)
        axis.errorbar(part.Marker_LSI_pct, y,
                      xerr=[part.Marker_LSI_pct - part.Marker_CI_low,
                            part.Marker_CI_high - part.Marker_LSI_pct],
                      fmt="s", color="black", capsize=3, label="Marker (95% CI)", markersize=5)
        axis.scatter(part.Isokinetic_LSI_pct, y + 0.18, marker="D", color="darkorange",
                     s=45, zorder=5, label="Isokinetic")
        axis.axvline(100, color="grey", ls="--", lw=1)
        axis.set_yticks(y)
        axis.set_yticklabels([f"[{g}] {m}" for g, m in zip(part.Group, part.Metric)], fontsize=8)
        axis.invert_yaxis()
        # Clip to a readable range: the knee-flexor CIs run to several
        # thousand percent, which otherwise squashes every usable metric into
        # one pixel. Off-scale bounds are marked rather than silently hidden.
        axis.set_xlim(0, X_LIMIT)
        for row in part.itertuples():
            for value, offset in ((row.Framework_CI_high, -0.18), (row.Marker_CI_high, 0.0)):
                if np.isfinite(value) and value > X_LIMIT:
                    axis.annotate("", xy=(X_LIMIT, row.Index + offset),
                                  xytext=(X_LIMIT * 0.94, row.Index + offset),
                                  arrowprops=dict(arrowstyle="->", color="grey", lw=1.2))
                    axis.text(X_LIMIT * 0.955, row.Index + offset - 0.12,
                              f"{value:.0f}", fontsize=6, color="grey")
        axis.set_xlabel("LSI (%)  -- arrows mark CI upper bounds beyond the axis")
        axis.set_title(f"AB03 {comparison}")
        axis.grid(True, alpha=0.3, axis="x")
        axis.legend(fontsize=7, loc="lower right")
    figure.tight_layout()
    figure.savefig(out / "ab03_lsi_association.png", dpi=140)
    plt.close(figure)
    print(f"-> {out / 'ab03_lsi_association.png'}")


if __name__ == "__main__":
    main()
