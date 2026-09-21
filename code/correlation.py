"""CorrelationAnalysis: SLJ-derived LSI vs. the gold-standard isokinetic LSI.

This is the actual final deliverable of the project -- does the cheap
framework (markerless + insole) track the isokinetic gold standard as well
as the lab-grade marker system does?

The isokinetic condition to join against comes from each LSI definition's
*reference* condition (config.lsi_definitions), mapped through
config.isokinetic_condition_map -- e.g. "NOEXO_L_vs_R" has reference
condition NOEXO -> isokinetic "NOEXO"; "NOEXO_L_vs_EXO_R" has reference
condition EXO -> isokinetic "WITHEXO". This is derived from config, not a
second hardcoded mapping.

SLJ metric -> isokinetic metric is a real, if inevitably approximate,
biomechanical analogy: peak knee (extensor) moment during landing is the
closest SLJ correlate of isokinetic peak quad torque; knee absolute work is
the closest correlate of isokinetic total work.

Every output reports n explicitly. Never assume a fixed subject count --
the join is generic and grows as more subjects gain both datasets.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import DEFAULT
from isokinetic import IsokineticTable
from lsi import LsiCalculator

SLJ_TO_ISOKINETIC_METRIC = {
    "PeakKneeMoment": "peak_torque",
    "KneeAbsoluteWork": "total_work",
}


class CorrelationAnalysis:
    def __init__(self, config=DEFAULT):
        self.config = config
        self.lsi_calculator = LsiCalculator(config)
        self.isokinetic = IsokineticTable(config)

    def _lsi_summary_for(self, subject, activity):
        path = self.lsi_calculator.lsi_summary_path(subject, activity)
        if not path.is_file():
            return None
        return pd.read_csv(path)

    def build_join(self, subjects, activity) -> pd.DataFrame:
        """One row per (subject, LSI comparison, metric) with all three LSI values."""
        iso_long = self.isokinetic.load_long()
        rows = []
        for subject in subjects:
            lsi_summary = self._lsi_summary_for(subject, activity)
            if lsi_summary is None:
                continue
            for name, _test_condition, _test_side, reference_condition, _reference_side in self.config.lsi_definitions:
                iso_condition = self.config.isokinetic_condition_map.get(reference_condition)
                if iso_condition is None:
                    continue
                for slj_metric, iso_metric in SLJ_TO_ISOKINETIC_METRIC.items():
                    lsi_row = lsi_summary[
                        (lsi_summary.Comparison == name) & (lsi_summary.Metric == slj_metric)
                    ]
                    iso_row = iso_long[
                        (iso_long.subject == subject.id)
                        & (iso_long.condition == iso_condition)
                        & (iso_long.metric == iso_metric)
                    ]
                    if len(lsi_row) != 1 or len(iso_row) != 1:
                        continue
                    rows.append({
                        "subject": subject.id, "comparison": name,
                        "slj_metric": slj_metric, "iso_metric": iso_metric,
                        "framework_lsi": float(lsi_row["Framework_pct"].iloc[0]),
                        "marker_lsi": float(lsi_row["Marker_pct"].iloc[0]),
                        "isokinetic_lsi": float(iso_row["iso_lsi"].iloc[0]),
                        "slj_test_trials": int(lsi_row["TestTrials"].iloc[0]),
                        "slj_reference_trials": int(lsi_row["ReferenceTrials"].iloc[0]),
                    })
        return pd.DataFrame(rows)

    @staticmethod
    def compute_statistics(joined: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for (comparison, slj_metric), group in joined.groupby(["comparison", "slj_metric"]):
            n = len(group)
            for pair_name, column in (
                ("framework_vs_isokinetic", "framework_lsi"),
                ("marker_vs_isokinetic", "marker_lsi"),
            ):
                pearson_r = group[column].corr(group["isokinetic_lsi"], method="pearson") if n >= 2 else float("nan")
                spearman_r = group[column].corr(group["isokinetic_lsi"], method="spearman") if n >= 2 else float("nan")
                rows.append({
                    "comparison": comparison, "slj_metric": slj_metric, "pair": pair_name,
                    "n": n, "pearson_r": pearson_r, "spearman_r": spearman_r,
                })
        return pd.DataFrame(rows)

    @staticmethod
    def framework_vs_marker_agreement(joined: pd.DataFrame) -> pd.DataFrame:
        """Bland-Altman-style agreement between framework and marker SLJ LSI
        (independent of the isokinetic join -- useful even before comparing
        either one to the gold standard)."""
        rows = []
        for (comparison, slj_metric), group in joined.groupby(["comparison", "slj_metric"]):
            diff = group["framework_lsi"] - group["marker_lsi"]
            n = len(group)
            sd = diff.std(ddof=1) if n > 1 else float("nan")
            bias = diff.mean()
            rows.append({
                "comparison": comparison, "slj_metric": slj_metric, "n": n,
                "bias_pct_points": bias, "sd_pct_points": sd,
                "limits_of_agreement_low": bias - 1.96 * sd if n > 1 else float("nan"),
                "limits_of_agreement_high": bias + 1.96 * sd if n > 1 else float("nan"),
            })
        return pd.DataFrame(rows)

    def run(self, subjects, activity, make_plots: bool = True):
        joined = self.build_join(subjects, activity)
        out_dir = self.config.results_root / "_correlation"
        out_dir.mkdir(parents=True, exist_ok=True)

        joined.to_csv(out_dir / f"{activity.name}_slj_vs_isokinetic.csv", index=False)
        statistics = self.compute_statistics(joined)
        statistics.to_csv(out_dir / f"{activity.name}_correlation_stats.csv", index=False)
        agreement = self.framework_vs_marker_agreement(joined)
        agreement.to_csv(out_dir / f"{activity.name}_framework_vs_marker_agreement.csv", index=False)

        if make_plots and len(joined):
            self._plot(joined, out_dir, activity)
        return joined, statistics, agreement

    @staticmethod
    def _plot(joined: pd.DataFrame, out_dir: Path, activity):
        for (comparison, slj_metric), group in joined.groupby(["comparison", "slj_metric"]):
            figure, axis = plt.subplots(figsize=(6, 6))
            axis.scatter(group["isokinetic_lsi"], group["framework_lsi"], label="Framework", color="tab:red")
            axis.scatter(group["isokinetic_lsi"], group["marker_lsi"], label="Marker", color="tab:blue")
            axis.set_xlabel("Isokinetic LSI (%)")
            axis.set_ylabel("SLJ LSI (%)")
            axis.set_title(f"{comparison} | {slj_metric} | n={len(group)}")
            axis.legend()
            figure.tight_layout()
            figure.savefig(out_dir / f"{activity.name}_{comparison}_{slj_metric}_scatter.png", dpi=160)
            plt.close(figure)

    def subject_report_dir(self, subject, activity) -> Path:
        return self.config.results_root / subject.id / "analysis" / activity.name / "isokinetic_association"

    def subject_report(self, subject, activity, make_plots: bool = True) -> pd.DataFrame:
        """This ONE subject's own SLJ LSI vs. their isokinetic LSI, for every
        condition (NOEXO, EXO) -- a per-subject companion to the aggregate,
        cross-subject `run()`/results/_correlation/ output. Reuses build_join
        (single source of truth for the join logic), just scoped to one
        subject and written into that subject's own analysis/ folder instead
        of the aggregate one.

        Writes an empty (header-only) CSV rather than nothing when the
        subject has no isokinetic row (e.g. NTS01) -- so a human opening the
        folder sees *why* there's no association, not a missing file.
        """
        out_dir = self.subject_report_dir(subject, activity)
        out_dir.mkdir(parents=True, exist_ok=True)
        joined = self.build_join([subject], activity)
        csv_path = out_dir / f"{activity.name}_slj_vs_isokinetic.csv"
        joined.to_csv(csv_path, index=False)

        if joined.empty:
            print(f"  {subject.id}: no isokinetic data available -- wrote header-only {csv_path.name}")
            return joined

        if make_plots:
            self._plot_subject(subject, joined, out_dir, activity)
        return joined

    @staticmethod
    def _plot_subject(subject, joined: pd.DataFrame, out_dir: Path, activity):
        """One subject's framework/marker/isokinetic LSI side by side, per
        (condition comparison, metric) -- a bar chart, not a scatter (a
        single subject has no cross-subject spread to plot against)."""
        labels = [f"{row.comparison}\n{row.slj_metric}" for row in joined.itertuples()]
        figure, axis = plt.subplots(figsize=(max(6, 2.2 * len(joined)), 6))
        positions = range(len(joined))
        width = 0.25
        axis.bar([p - width for p in positions], joined["framework_lsi"], width, label="Framework", color="tab:red")
        axis.bar(list(positions), joined["marker_lsi"], width, label="Marker", color="tab:blue")
        axis.bar([p + width for p in positions], joined["isokinetic_lsi"], width, label="Isokinetic", color="tab:green")
        axis.axhline(100, color="k", linestyle="--", linewidth=1)
        axis.set_xticks(list(positions))
        axis.set_xticklabels(labels, fontsize=8)
        axis.set_ylabel("LSI (%)")
        axis.set_title(f"{subject.id} | SLJ vs. isokinetic LSI")
        axis.legend()
        figure.tight_layout()
        figure.savefig(out_dir / f"{activity.name}_slj_vs_isokinetic.png", dpi=160)
        plt.close(figure)
