"""analyze_screening.py -- which kinematics-only predictor actually tells you
a trial can support an Fy (vertical) gain estimate?

Reads results/trial_metrics.csv (from collect_metrics.py), ranks every
candidate predictor by how well it explains the ground-truth outcome, and
proposes/scores a concrete screening rule.

Outcome variable: `DeltaRMSE_active_N` = RMSE(pure COM-translational) -
RMSE(COM+H-dot blend), over frames where the H-dot term is actually active.
Positive => the H-dot term genuinely improved on pure translational there.
This deliberately is NOT raw RMSE: a trial can post a good whole-trace RMSE
purely because the lever mask bypassed the H-dot term and fell back to
translational (which equals body weight at rest and matches the plate).

Run from code/:
    ..\\opensimPy\\Scripts\\python.exe expt_temp/fy_gain_trial_screening/analyze_screening.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
PLOTS = RESULTS / "plots"

PREDICTORS = [
    "LeverX_abs_median_m", "LeverX_abs_p75_m", "LeverX_abs_max_m",
    "FracLeverX_gt_05", "FracLeverX_gt_10", "FracLeverX_gt_15",
    "Amplification_median", "Amplification_p25",
    "Identifiability_median", "Identifiability_p75",
    "ActiveFraction", "ActiveCrossings",
    "Mz_abs_median_Nm", "ContactDuration_s", "ContactBouts",
]
OUTCOME = "DeltaRMSE_active_N"


def main():
    table = pd.read_csv(RESULTS / "trial_metrics.csv")
    usable = table[np.isfinite(table[OUTCOME])].copy()
    print(f"{len(table)} trials collected, {len(usable)} with a valid outcome\n")

    helped = usable[OUTCOME] > 0
    print(f"H-dot term improved on pure translational in {helped.sum()}/{len(usable)} trials "
          f"({100 * helped.mean():.0f}%)")
    print(f"  median DeltaRMSE_active: {usable[OUTCOME].median():+.1f} N")
    print(f"  median RMSE blend / translational (active frames): "
          f"{usable['RMSE_blend_active_N'].median():.0f} / "
          f"{usable['RMSE_translational_active_N'].median():.0f} N\n")

    print(f"{'predictor':<26} {'spearman_rho':>13} {'p':>10}   vs " + OUTCOME)
    print("-" * 68)
    ranking = []
    for name in PREDICTORS:
        if name not in usable.columns:
            continue
        pair = usable[[name, OUTCOME]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 10 or pair[name].nunique() < 3:
            continue
        rho, pvalue = stats.spearmanr(pair[name], pair[OUTCOME])
        ranking.append((name, rho, pvalue, len(pair)))
    for name, rho, pvalue, count in sorted(ranking, key=lambda r: -abs(r[1])):
        flag = " *" if pvalue < 0.05 else ""
        print(f"{name:<26} {rho:>13.3f} {pvalue:>10.4f}   (n={count}){flag}")

    PLOTS.mkdir(parents=True, exist_ok=True)
    _scatter_grid(usable, ranking)
    _rule_report(usable)


def _scatter_grid(usable, ranking):
    ordered = [name for name, _, _, _ in sorted(ranking, key=lambda r: -abs(r[1]))]
    columns = 3
    rows = int(np.ceil(len(ordered) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    subjects = sorted(usable["Subject"].unique())
    colors = dict(zip(subjects, plt.cm.tab10(np.linspace(0, 1, max(len(subjects), 2)))))

    for axis, name in zip(axes, ordered):
        for subject in subjects:
            part = usable[usable["Subject"] == subject]
            axis.scatter(part[name].replace([np.inf, -np.inf], np.nan), part[OUTCOME],
                         s=22, alpha=0.8, color=colors[subject], label=subject)
        axis.axhline(0, color="black", linewidth=0.8, linestyle="--")
        pair = usable[[name, OUTCOME]].replace([np.inf, -np.inf], np.nan).dropna()
        rho, pvalue = stats.spearmanr(pair[name], pair[OUTCOME]) if len(pair) >= 10 else (np.nan, np.nan)
        axis.set_title(f"{name}\nrho={rho:.2f}, p={pvalue:.3f}", fontsize=9)
        axis.set_ylabel("DeltaRMSE active (N)", fontsize=8)
        axis.grid(True, alpha=0.3)
    for axis in axes[len(ordered):]:
        axis.axis("off")
    axes[0].legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(PLOTS / "predictors_vs_outcome.png", dpi=140)
    plt.close(figure)
    print(f"\nplot -> {PLOTS / 'predictors_vs_outcome.png'}")


def _rule_report(usable):
    """Score simple one-predictor threshold rules by how cleanly they separate
    trials the H-dot term helps from trials it hurts."""
    print(f"\n{'rule':<34} {'kept':>6} {'kept_helped':>12} {'precision':>10} {'recall':>8}")
    print("-" * 74)
    helped = usable[OUTCOME] > 0
    candidates = []
    for name in ("FracLeverX_gt_10", "FracLeverX_gt_15", "LeverX_abs_median_m",
                 "Identifiability_median", "ActiveFraction"):
        if name not in usable.columns:
            continue
        values = usable[name].replace([np.inf, -np.inf], np.nan)
        for quantile in (0.4, 0.5, 0.6, 0.7):
            threshold = values.quantile(quantile)
            if not np.isfinite(threshold):
                continue
            keep = values >= threshold
            if keep.sum() < 5:
                continue
            precision = float(helped[keep].mean())
            recall = float(keep[helped].mean()) if helped.any() else np.nan
            candidates.append((f"{name} >= {threshold:.3f}", int(keep.sum()),
                               int((keep & helped).sum()), precision, recall))
    for label, kept, kept_helped, precision, recall in sorted(candidates, key=lambda r: -r[3]):
        print(f"{label:<34} {kept:>6} {kept_helped:>12} {precision:>10.2f} {recall:>8.2f}")
    print(f"\nbaseline (keep everything): precision {helped.mean():.2f}")


if __name__ == "__main__":
    main()
