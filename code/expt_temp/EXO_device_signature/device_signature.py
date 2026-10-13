"""device_signature.py -- can the SLJ pipelines detect the exoskeleton's
direction-specific effect that the isokinetic test measures?

The signal
----------
In `EXO_R_vs_NOEXO_R` (same limb, device on vs off) the isokinetic reference
shows a consistent, DIRECTIONAL effect in every subject tested: the device
raises knee EXTENSION and lowers knee FLEXION.

    AB02 +13.7   AB03 +12.7   AB04 +24.1   AB05 +18.5
    AB07 +12.5   AB08 +40.1   AB09 +41.0        (extension mean - flexion mean)

Positive in 7/7. This is the only comparison in the dataset carrying a large,
direction-specific signal -- NOEXO_L_vs_R is near-symmetric (mean deviation
from 100 of 2.8 points, versus 6.0 here), which is why testing specificity on
it was underpowered.

The test
--------
Within-limb, so between-limb anatomy cancels. For each subject the SLJ metrics
are grouped by which isokinetic group their mechanism predicts, and the SLJ
"divergence" (mean extension-metric LSI minus mean flexion-metric LSI) is
compared with the isokinetic divergence.

  * SIGN test -- isokinetic divergence is positive for every subject, so a
    pipeline that detects the effect should also be positive. 4/4 would be
    p = 0.0625 one-tailed; it cannot reach 0.05 at n=4, which is stated rather
    than glossed.
  * MAGNITUDE -- does the subject with the largest isokinetic divergence
    (AB08, +40.1) also show the largest SLJ divergence?
  * PER-METRIC -- which single SLJ metric tracks the isokinetic extension and
    flexion LSIs best across subjects.

Mechanism mapping (see AB03_FullData_Iso_association/README.md): quadriceps
route via knee/ankle extensor and support moment; hamstring route via HIP
extensor, because hamstrings are biarticular and hip extension is their jump
role -- the knee flexor moment in a landing is noise.

Run from code/:
    ..\\opensimPy\\Scripts\\python.exe expt_temp/EXO_device_signature/device_signature.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

CODE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(CODE_ROOT))

from config import DEFAULT  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
WORKBOOK = DEFAULT.data_root / "Isokinetic_Study_AB02_to_AB09_VALIDATED.xlsx"
SUBJECT_IDS = ("AB03", "AB05", "AB07", "AB08")
BOOTSTRAP = 5000
RANDOM_SEED = 20260930

ISO_METRICS = ["PeakTorque_ftlb", "TotalWork_ftlb", "AvgPower_W"]

# (movement stem, label, predicted isokinetic group)
METRICS = [
    ("PushOff_KneeConcWork_J", "Push-off knee concentric work", "Extension"),
    ("PushOff_KneeExtPeak_Nm", "Push-off knee extensor peak", "Extension"),
    ("PushOff_AnkleConcWork_J", "Push-off ankle concentric work", "Extension"),
    ("PushOff_SupportPeak_Nm", "Push-off support moment peak", "Extension"),
    ("Landing_KneeEccWork_J", "Landing knee eccentric work", "Extension"),
    ("Landing_KneeExtPeak_Nm", "Landing knee extensor peak", "Extension"),
    ("Landing_SupportPeak_Nm", "Landing support moment peak", "Extension"),
    ("PushOff_HipConcWork_J", "Push-off hip concentric work", "Flexion"),
    ("PushOff_HipExtPeak_Nm", "Push-off hip extensor peak", "Flexion"),
    ("Landing_HipEccWork_J", "Landing hip eccentric work", "Flexion"),
]


def _lsi_with_ci(test_values, reference_values, rng):
    test_values = np.asarray(test_values, float)
    reference_values = np.asarray(reference_values, float)
    test_values = test_values[np.isfinite(test_values)]
    reference_values = reference_values[np.isfinite(reference_values)]
    if len(test_values) < 2 or len(reference_values) < 2 or reference_values.mean() == 0:
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
    kinetics = pd.read_csv(RESULTS / "phase_kinetics.csv")
    iso = pd.read_excel(WORKBOOK, "Isokinetic_LSI")
    iso = iso[iso.Comparison == "EXO_R_vs_NOEXO_R"]
    rng = np.random.default_rng(RANDOM_SEED)

    iso_rows = []
    for subject_id in SUBJECT_IDS:
        part = iso[iso.Subject == subject_id]
        for group in ("Extension", "Flexion"):
            values = part[part.Metric.isin([f"{group}_{m}" for m in ISO_METRICS])].LSI_pct
            iso_rows.append({"Subject": subject_id, "Group": group,
                             "Iso_LSI_pct": round(float(values.mean()), 1) if len(values) else np.nan,
                             "n_metrics": len(values)})
    iso_table = pd.DataFrame(iso_rows)
    iso_wide = iso_table.pivot(index="Subject", columns="Group", values="Iso_LSI_pct")
    iso_wide["Iso_Divergence"] = (iso_wide.Extension - iso_wide.Flexion).round(1)

    rows = []
    for subject_id in SUBJECT_IDS:
        data = kinetics[kinetics.Subject == subject_id]
        test = data[(data.Condition == "EXO") & (data.Side == "R")]
        reference = data[(data.Condition == "NOEXO") & (data.Side == "R")]
        if not len(test) or not len(reference):
            continue
        for stem, label, predicted in METRICS:
            record = {"Subject": subject_id, "Metric": label, "Predicted": predicted,
                      "TestTrials": len(test), "ReferenceTrials": len(reference)}
            for pipeline in ("Framework", "Marker"):
                column = f"{pipeline}_{stem}"
                if column not in kinetics.columns:
                    continue
                point, low, high = _lsi_with_ci(test[column], reference[column], rng)
                record[f"{pipeline}_LSI_pct"] = round(point, 1) if np.isfinite(point) else np.nan
                record[f"{pipeline}_CI_low"] = round(low, 1) if np.isfinite(low) else np.nan
                record[f"{pipeline}_CI_high"] = round(high, 1) if np.isfinite(high) else np.nan
            rows.append(record)
    table = pd.DataFrame(rows)

    # The marker ankle concentric work metric was already found unreliable on
    # AB03 (LSI 320-340%). Check whether that repeats before trusting it.
    ankle = table[table.Metric == "Push-off ankle concentric work"]
    extreme = ankle[(ankle.Marker_LSI_pct > 200) | (ankle.Marker_LSI_pct < 50)]
    if len(extreme):
        print("=== EXCLUDED: marker push-off ankle concentric work is unreliable ===")
        print(extreme[["Subject", "Framework_LSI_pct", "Marker_LSI_pct"]].to_string(index=False))
        print()
    table = table[table.Metric != "Push-off ankle concentric work"]

    table.to_csv(RESULTS / "device_signature_per_metric.csv", index=False)

    summary = []
    for subject_id in SUBJECT_IDS:
        part = table[table.Subject == subject_id]
        record = {"Subject": subject_id,
                  "Iso_Extension": iso_wide.loc[subject_id, "Extension"],
                  "Iso_Flexion": iso_wide.loc[subject_id, "Flexion"],
                  "Iso_Divergence": iso_wide.loc[subject_id, "Iso_Divergence"]}
        for pipeline in ("Framework", "Marker"):
            extension = part[part.Predicted == "Extension"][f"{pipeline}_LSI_pct"]
            flexion = part[part.Predicted == "Flexion"][f"{pipeline}_LSI_pct"]
            record[f"{pipeline}_Extension"] = round(float(extension.mean()), 1)
            record[f"{pipeline}_Flexion"] = round(float(flexion.mean()), 1)
            record[f"{pipeline}_Divergence"] = round(
                float(extension.mean() - flexion.mean()), 1)
        summary.append(record)
    summary = pd.DataFrame(summary)
    summary.to_csv(RESULTS / "device_signature_summary.csv", index=False)

    pd.set_option("display.width", 220)
    print("=== per-subject divergence: extension-metric mean minus flexion-metric mean ===")
    print(summary[["Subject", "Iso_Extension", "Iso_Flexion", "Iso_Divergence",
                   "Framework_Extension", "Framework_Flexion", "Framework_Divergence",
                   "Marker_Extension", "Marker_Flexion", "Marker_Divergence"]].to_string(index=False))

    print("\n=== SIGN test (isokinetic divergence is positive for all 4) ===")
    for pipeline in ("Framework", "Marker"):
        values = summary[f"{pipeline}_Divergence"]
        positive = int((values > 0).sum())
        print(f"  {pipeline}: positive in {positive}/{len(values)} subjects "
              f"(mean {values.mean():+.1f}, isokinetic mean {summary.Iso_Divergence.mean():+.1f})")
    print("  note: 4/4 would be p=0.0625 one-tailed -- n=4 cannot reach 0.05 by sign test alone")

    print("\n=== MAGNITUDE: does SLJ divergence track isokinetic divergence? ===")
    for pipeline in ("Framework", "Marker"):
        x = summary.Iso_Divergence.to_numpy(float)
        y = summary[f"{pipeline}_Divergence"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 3:
            rho, p = stats.spearmanr(x[ok], y[ok])
            r, pr = stats.pearsonr(x[ok], y[ok])
            print(f"  {pipeline}: Spearman rho={rho:+.2f} (p={p:.3f}), "
                  f"Pearson r={r:+.2f} (p={pr:.3f}), n={ok.sum()}")
    print("  note: at n=4, |r| must exceed 0.95 for p<0.05 -- treat as descriptive")

    print("\n=== PER-METRIC: which SLJ metric tracks the isokinetic group across subjects? ===")
    merged = table.merge(iso_wide.reset_index(), on="Subject")
    out = []
    for (label, predicted), part in merged.groupby(["Metric", "Predicted"]):
        target = part.Extension if predicted == "Extension" else part.Flexion
        record = {"Metric": label, "Predicted": predicted, "n": len(part)}
        for pipeline in ("Framework", "Marker"):
            values = part[f"{pipeline}_LSI_pct"]
            ok = np.isfinite(values) & np.isfinite(target)
            record[f"{pipeline}_MeanAbsGap"] = round(float(np.abs(values[ok] - target[ok]).mean()), 1)
            if ok.sum() >= 3:
                record[f"{pipeline}_rho"] = round(float(stats.spearmanr(values[ok], target[ok])[0]), 2)
        out.append(record)
    per_metric = pd.DataFrame(out).sort_values("Framework_MeanAbsGap")
    per_metric.to_csv(RESULTS / "device_signature_per_metric_ranking.csv", index=False)
    print(per_metric.to_string(index=False))

    _plot(summary, table, iso_wide)
    print(f"\n-> {RESULTS}")


def _plot(summary, table, iso_wide):
    out = RESULTS / "plots"
    out.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.6))

    axis = axes[0]
    x = np.arange(len(summary))
    width = 0.26
    axis.bar(x - width, summary.Iso_Divergence, width, color="darkorange", label="Isokinetic")
    axis.bar(x, summary.Framework_Divergence, width, color="red", label="Framework")
    axis.bar(x + width, summary.Marker_Divergence, width, color="black", label="Marker")
    axis.axhline(0, color="grey", lw=1)
    axis.set_xticks(x)
    axis.set_xticklabels(summary.Subject)
    axis.set_ylabel("divergence: extension LSI - flexion LSI (points)")
    axis.set_title("Device signature: isokinetic is positive in every subject")
    axis.grid(True, alpha=0.3, axis="y")
    axis.legend(fontsize=8)

    axis = axes[1]
    for pipeline, colour, marker in (("Framework", "red", "o"), ("Marker", "black", "s")):
        axis.scatter(summary.Iso_Divergence, summary[f"{pipeline}_Divergence"],
                     color=colour, marker=marker, s=55, label=pipeline)
        for _, row in summary.iterrows():
            axis.annotate(row.Subject, (row.Iso_Divergence, row[f"{pipeline}_Divergence"]),
                          fontsize=7, xytext=(4, 3), textcoords="offset points")
    limits = [-40, 60]
    axis.plot(limits, limits, ls="--", color="grey", lw=1, label="perfect agreement")
    axis.axhline(0, color="grey", lw=0.8)
    axis.axvline(0, color="grey", lw=0.8)
    axis.set_xlabel("isokinetic divergence (points)")
    axis.set_ylabel("SLJ divergence (points)")
    axis.set_title("Does SLJ track the device effect? (n=4, descriptive)")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(out / "device_signature.png", dpi=140)
    plt.close(figure)
    print(f"-> {out / 'device_signature.png'}")


if __name__ == "__main__":
    main()
