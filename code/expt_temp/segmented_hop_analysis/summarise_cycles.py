"""summarise_cycles.py -- counts, symmetry profile, LSI and sync diagnostics
from segment_hops.py's per-cycle output.

Produces:
  results/cycle_counts.csv   cycles + accepted attempts per subject/activity/
                             condition/side, plus a COUNT-based symmetry ratio
                             (the "own symmetry profile" for successful attempts)
  results/lsi_summary.csv    LSI per subject/activity via the production
                             LsiCalculator, so the definitions and the
                             100*mean(test)/mean(reference) convention are
                             identical to SLJ's -- not a parallel reimplementation
  results/sync_quality.csv   per trial: cross-correlation vs peak-matching
                             offset and their disagreement
  results/plots/summary_counts.png

Only ACCEPTED cycles feed the LSI. For LH that means cycles clearing the
40 cm ML landing-separation gate; for SKH every complete left->right->left
cycle is accepted (the gate is LH-specific by definition).

Joint-moment and COP metrics are absent by design in this pass -- they need
the inverse-dynamics stage over the widened whole-bout window, deferred until
the segmentation itself is confirmed. Their LSI rows therefore come out NaN
rather than being silently dropped.

Run from code/:
    ..\\opensimPy\\Scripts\\python.exe expt_temp/segmented_hop_analysis/summarise_cycles.py
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
from lsi import METRICS, LsiCalculator  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"


def main():
    table = pd.read_csv(RESULTS / "cycle_metrics.csv")
    print(f"{len(table)} stance rows, {table.Trial.nunique()} trials, "
          f"{table.Subject.nunique()} subjects\n")

    # ---- counts + count-based symmetry profile -----------------------
    counts = (table.groupby(["Subject", "Activity", "Condition", "Side"])
              .agg(Stances=("Cycle", "size"),
                   AcceptedStances=("Accepted", "sum"),
                   Trials=("Trial", "nunique"))
              .reset_index())
    counts["AcceptedStances"] = counts.AcceptedStances.astype(int)
    counts.to_csv(RESULTS / "cycle_counts.csv", index=False)
    print("=== accepted stances per subject / activity / condition / side ===")
    pivot = counts.pivot_table(index=["Subject", "Activity"],
                               columns=["Condition", "Side"],
                               values="AcceptedStances", aggfunc="sum").fillna(0).astype(int)
    print(pivot.to_string())

    # Count symmetry: 100 * accepted(NOEXO,L) / accepted(NOEXO,R), and the
    # EXO-R variant, matching config.lsi_definitions' own pairings.
    print("\n=== count-based symmetry profile (successful attempts) ===")
    rows = []
    for (subject, activity), group in counts.groupby(["Subject", "Activity"]):
        def accepted(condition, side):
            hit = group[(group.Condition == condition) & (group.Side == side)]
            return int(hit.AcceptedStances.sum())
        for name, test_condition, test_side, reference_condition, reference_side in DEFAULT.lsi_definitions:
            test, reference = accepted(test_condition, test_side), accepted(reference_condition, reference_side)
            rows.append({
                "Subject": subject, "Activity": activity, "Comparison": name,
                "TestAccepted": test, "ReferenceAccepted": reference,
                "CountSymmetry_pct": round(100 * test / reference, 1) if reference else np.nan,
            })
    count_symmetry = pd.DataFrame(rows)
    count_symmetry.to_csv(RESULTS / "count_symmetry.csv", index=False)
    print(count_symmetry.to_string(index=False))

    # ---- LSI via the production calculator ---------------------------
    accepted_only = table[table.Accepted].copy()
    for _category, _metric, framework_column, marker_column in METRICS:
        for column in (framework_column, marker_column):
            if column not in accepted_only.columns:
                accepted_only[column] = np.nan
    calculator = LsiCalculator(DEFAULT)
    lsi_rows = []
    for (subject, activity), group in accepted_only.groupby(["Subject", "Activity"]):
        ratios = calculator.compute(group)
        ratios.insert(0, "Activity", activity)
        ratios.insert(0, "Subject", subject)
        lsi_rows.append(ratios)
    lsi = pd.concat(lsi_rows, ignore_index=True)
    lsi.to_csv(RESULTS / "lsi_summary.csv", index=False)
    print("\n=== LSI (PeakGRF; other metrics NaN pending inverse dynamics) ===")
    grf = lsi[lsi.Metric == "PeakGRF"].drop(columns=["OutcomeType", "Metric"])
    print(grf.to_string(index=False))

    # ---- sync diagnostics --------------------------------------------
    sync = (table.groupby(["Subject", "Activity", "Trial"])
            .agg(CrossCorr_s=("SyncCrossCorr_s", "first"),
                 Peak_s=("SyncPeak_s", "first"),
                 Disagreement_s=("SyncDisagreement_s", "first"))
            .reset_index().sort_values("Disagreement_s", ascending=False))
    sync.to_csv(RESULTS / "sync_quality.csv", index=False)
    print("\n=== sync: cross-correlation vs peak-matching disagreement ===")
    print(f"median {sync.Disagreement_s.median():.3f}s | "
          f">1s in {(sync.Disagreement_s > 1).sum()} of {len(sync)} trials")
    print(sync.head(10).to_string(index=False))

    # ---- GRF accuracy by activity ------------------------------------
    print("\n=== GRF RMSE over accepted stances ===")
    print(accepted_only.groupby("Activity")[["GrfRmse_N", "GrfXRmse_N", "GrfZRmse_N"]]
          .median().round(0).to_string())

    _plot_counts(counts)


def _plot_counts(counts):
    out = RESULTS / "plots"
    out.mkdir(parents=True, exist_ok=True)
    activities = sorted(counts.Activity.unique())
    figure, axes = plt.subplots(1, len(activities), figsize=(7 * len(activities), 4.6), squeeze=False)
    for index, activity in enumerate(activities):
        axis = axes[0][index]
        part = counts[counts.Activity == activity]
        pivot = part.pivot_table(index="Subject", columns=["Condition", "Side"],
                                 values="AcceptedStances", aggfunc="sum").fillna(0)
        pivot.plot(kind="bar", ax=axis, width=0.8)
        axis.set_title(f"{activity}: accepted stances per subject")
        axis.set_ylabel("accepted stances")
        axis.grid(True, alpha=0.3, axis="y")
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(out / "summary_counts.png", dpi=140)
    plt.close(figure)
    print(f"\nplot -> {out / 'summary_counts.png'}")


if __name__ == "__main__":
    main()
