"""LsiCalculator: Limb Symmetry Index from a trial_summary DataFrame.

100 * mean(test group) / mean(reference group), directional per the study
convention (so LSI can exceed 100%). Comparison groups come from
config.lsi_definitions, not a hardcoded axis, so the comparison groups can be
changed in one place without touching this module. Verified against
Isokinetic_Data.xlsx's own LSI columns (see CLAUDE.md's Domain facts): same
formula, same direction, for the same NOEXO_L_vs_R / NOEXO_L_vs_EXO_R axes.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import DEFAULT

METRICS = [
    ("Traditional", "JumpHeight", "FrameworkJumpHeight_m", "MarkerJumpHeight_m"),
    ("Functional/Biomechanical", "PeakGRF", "PeakFrameworkGRF_N", "PeakMarkerGRF_N"),
    ("Functional/Biomechanical", "PeakHipMoment", "PeakFrameworkHip_Nm", "PeakMarkerHip_Nm"),
    ("Functional/Biomechanical", "PeakKneeMoment", "PeakFrameworkKnee_Nm", "PeakMarkerKnee_Nm"),
    ("Functional/Biomechanical", "PeakAnkleMoment", "PeakFrameworkAnkle_Nm", "PeakMarkerAnkle_Nm"),
    ("Functional/Biomechanical", "PeakSupportMoment", "PeakFrameworkSupportMoment_Nm", "PeakMarkerSupportMoment_Nm"),
    ("Functional/Biomechanical", "HipAbsoluteWork", "FrameworkHipAbsoluteWork_J", "MarkerHipAbsoluteWork_J"),
    ("Functional/Biomechanical", "KneeAbsoluteWork", "FrameworkKneeAbsoluteWork_J", "MarkerKneeAbsoluteWork_J"),
    ("Functional/Biomechanical", "AnkleAbsoluteWork", "FrameworkAnkleAbsoluteWork_J", "MarkerAnkleAbsoluteWork_J"),
    ("Functional/Biomechanical", "TotalAbsoluteWork", "FrameworkTotalAbsoluteWork_J", "MarkerTotalAbsoluteWork_J"),
]
LSI_COLUMNS = ["Comparison", "OutcomeType", "Metric", "Framework_pct", "Marker_pct", "TestTrials", "ReferenceTrials"]


class LsiCalculator:
    def __init__(self, config=DEFAULT):
        self.config = config

    def lsi_summary_path(self, subject, activity) -> Path:
        return self.config.results_root / subject.id / "analysis" / activity.name / "lsi_summary.csv"

    def compute(self, summary: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for name, test_condition, test_side, reference_condition, reference_side in self.config.lsi_definitions:
            test = summary[(summary.Condition == test_condition) & (summary.Side == test_side)]
            reference = summary[(summary.Condition == reference_condition) & (summary.Side == reference_side)]
            for category, metric, framework_column, marker_column in METRICS:
                framework_test = test[framework_column].mean()
                framework_reference = reference[framework_column].mean()
                marker_test = test[marker_column].mean()
                marker_reference = reference[marker_column].mean()
                rows.append([
                    name, category, metric,
                    100 * framework_test / framework_reference,
                    100 * marker_test / marker_reference,
                    len(test), len(reference),
                ])
        return pd.DataFrame(rows, columns=LSI_COLUMNS)

    def run(self, subject, activity, summary: pd.DataFrame, make_plots: bool = True) -> pd.DataFrame:
        ratios = self.compute(summary)
        out = self.lsi_summary_path(subject, activity)
        out.parent.mkdir(parents=True, exist_ok=True)
        ratios.to_csv(out, index=False)
        if make_plots:
            self._plot(ratios, out.parent)
        return ratios

    @staticmethod
    def _plot(ratios: pd.DataFrame, out_dir: Path):
        for name, group in ratios.groupby("Comparison"):
            figure, axis = plt.subplots(figsize=(11, 6))
            group.plot.bar(x="Metric", y=["Framework_pct", "Marker_pct"], ax=axis)
            axis.axhline(100, color="k", linestyle="--")
            axis.set_ylabel("Test / reference ratio (%)")
            axis.set_title(name.replace("_", " "))
            figure.tight_layout()
            figure.savefig(out_dir / f"{name}_LSI.png", dpi=180)
            plt.close(figure)
