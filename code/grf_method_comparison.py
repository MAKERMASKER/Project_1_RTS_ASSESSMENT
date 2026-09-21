"""GrfMethodComparison: Fx/Fy/Fz framework-GRF diagnostic, translation-only
vs. H-dot-blended, both against the marker/force-plate ground truth.

This is a diagnostic/investigation tool, not part of the production
pipeline (nothing in project.py/cli.py calls it) -- it exists to answer a
specific question: does blending the translation-only COM-double-
differentiation horizontal force estimate with the angular-momentum-rate
(H-dot) moment-balance estimate (framework_processor.py's
`rotation_force_weight_x`/`_z`) actually improve Fx/Fy/Fz agreement with
the marker/force-plate ground truth, trial by trial?

It compares two FrameworkProcessor *runs* of the same subject/trials,
captured at two different times as two different `trial_data.npz` sets:
  - "translation-only": a subject_rules snapshot with no rotation-force
    blend (rotation_force_weight_x/z == 0, or the key absent) -- read from
    wherever that run's output was preserved (e.g. a `dump/` backup).
  - "hdot": a subject_rules snapshot with the blend enabled -- read from
    the current `results/<SUBJECT>/framework/...` (i.e. after rerunning
    FrameworkProcessor with the new subject_rules/<ID>.json).
Both must have already been run (this module does not run the pipeline
itself); it only loads each run's `trial_data.npz` (time + grf_N, written
by FrameworkProcessor) plus the marker/force-plate ground truth.

Output: results/additional_performance_GRF_Checks/<SUBJECT>/GRF/ --
one Fx/Fy/Fz comparison plot per trial (three lines: translation-only,
H-dot blend, marker ground truth; RMS-to-marker in each panel's title)
plus one summary CSV across trials.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DEFAULT
from discovery import TrialDiscovery
from io_tables import read_storage

AXES = ("Fx", "Fy", "Fz")


def _rms(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values ** 2))) if len(values) else float("nan")


class GrfMethodComparison:
    def __init__(self, config=DEFAULT):
        self.config = config
        self.discovery = TrialDiscovery(config)

    def out_dir(self, subject) -> Path:
        return self.config.results_root / "additional_performance_GRF_Checks" / subject.id / "GRF"

    @staticmethod
    def _load_grf(framework_root: Path, trial_name: str):
        package = framework_root / "framework" / "single_leg_jump"
        matches = list(package.glob(f"*/{trial_name}/trial_data.npz"))
        if not matches:
            return None
        data = np.load(matches[0], allow_pickle=True)
        return data["time"], data["grf_N"]

    def _marker_grf(self, trial):
        force_plate = read_storage(trial.force_mot)
        plate = 1 if trial.side == "R" else 2
        marker = np.column_stack([
            force_plate[f"ground_force_{plate}_v{axis}"].to_numpy(float) for axis in "xyz"
        ])
        return force_plate.time.to_numpy(float), marker

    def compare_subject(self, subject, translation_only_root: Path, hdot_root: Path, make_plots: bool = True) -> pd.DataFrame:
        """translation_only_root / hdot_root: each a subject results root
        (i.e. the parent of that run's `framework/` folder) -- e.g. a
        `dump/AB03_backup_.../` snapshot and the live `results/AB03/`."""
        activity_name = "single_leg_jump"
        out_dir = self.out_dir(subject)
        out_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for condition in self.config.conditions:
            for trial in self.discovery.trials(subject, self._activity(), condition, for_analysis=True):
                translation = self._load_grf(translation_only_root, trial.name)
                hdot = self._load_grf(hdot_root, trial.name)
                if translation is None or hdot is None:
                    continue
                translation_time, translation_grf = translation
                hdot_time, hdot_grf = hdot
                marker_time, marker_grf = self._marker_grf(trial)

                shift = float(hdot_time[np.argmax(hdot_grf[:, 1])] - marker_time[np.argmax(np.abs(marker_grf[:, 1]))])
                common_time = hdot_time
                translation_on_common = np.column_stack([
                    np.interp(common_time, translation_time, translation_grf[:, axis], left=np.nan, right=np.nan)
                    for axis in range(3)
                ])
                marker_on_common = np.column_stack([
                    np.interp(common_time, marker_time + shift, marker_grf[:, axis], left=np.nan, right=np.nan)
                    for axis in range(3)
                ])

                rmse_translation = [_rms(translation_on_common[:, axis] - marker_on_common[:, axis]) for axis in range(3)]
                rmse_hdot = [_rms(hdot_grf[:, axis] - marker_on_common[:, axis]) for axis in range(3)]

                rows.append({
                    "Trial": trial.name, "Condition": trial.condition, "Side": trial.side,
                    "FxRmse_translation_N": rmse_translation[0], "FxRmse_hdot_N": rmse_hdot[0],
                    "FyRmse_translation_N": rmse_translation[1], "FyRmse_hdot_N": rmse_hdot[1],
                    "FzRmse_translation_N": rmse_translation[2], "FzRmse_hdot_N": rmse_hdot[2],
                    "PeakFyTranslation_N": float(np.nanmax(translation_on_common[:, 1])),
                    "PeakFyHdot_N": float(np.nanmax(hdot_grf[:, 1])),
                    "PeakFyMarker_N": float(np.nanmax(np.abs(marker_on_common[:, 1]))),
                })

                if make_plots:
                    self._plot_trial(
                        trial, common_time, translation_on_common, hdot_grf, marker_on_common,
                        rmse_translation, rmse_hdot, out_dir,
                    )

        summary = pd.DataFrame(rows)
        summary.to_csv(out_dir / f"{activity_name}_grf_method_comparison_summary.csv", index=False)
        if make_plots and len(summary):
            self._plot_summary(summary, out_dir, activity_name, subject.id)
        return summary

    @staticmethod
    def _activity():
        from activities import activity_for
        return activity_for("single_leg_jump", DEFAULT)

    @staticmethod
    def _plot_trial(trial, time, translation, hdot, marker, rmse_translation, rmse_hdot, out_dir: Path):
        x = time - time[0]
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for axis, label in enumerate(AXES):
            ax = axes[axis]
            ax.plot(x, marker[:, axis], "k", label="Marker (ground truth)", linewidth=1.5)
            ax.plot(x, translation[:, axis], "tab:blue", label="Translation-only", linewidth=1.1)
            ax.plot(x, hdot[:, axis], "tab:red", label="H-dot blend (0.5/0.5)", linewidth=1.1)
            ax.set_title(
                f"{label} | RMS translation={rmse_translation[axis]:.1f} N, hdot={rmse_hdot[axis]:.1f} N"
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Force (N)")
        axes[0].legend(fontsize=8, loc="best")
        figure.suptitle(f"{trial.name} -- Fx/Fy/Fz: translation-only vs. H-dot blend vs. marker")
        figure.tight_layout()
        figure.savefig(out_dir / f"{trial.name}_fx_fy_fz_comparison.png", dpi=160)
        plt.close(figure)

    @staticmethod
    def _plot_summary(summary: pd.DataFrame, out_dir: Path, activity_name: str, subject_id: str):
        figure, axes = plt.subplots(1, 3, figsize=(15, 5))
        labels = summary["Trial"].str.replace(f"{subject_id}_", "", regex=False)
        x = np.arange(len(summary))
        width = 0.35
        for axis, label in enumerate(AXES):
            ax = axes[axis]
            ax.bar(x - width / 2, summary[f"{label}Rmse_translation_N"], width, label="Translation-only", color="tab:blue")
            ax.bar(x + width / 2, summary[f"{label}Rmse_hdot_N"], width, label="H-dot blend", color="tab:red")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=90, fontsize=7)
            ax.set_ylabel("RMS vs. marker (N)")
            ax.set_title(f"{label} RMS by trial")
        axes[0].legend(fontsize=8)
        figure.suptitle("Translation-only vs. H-dot blend: per-trial GRF RMS vs. marker ground truth")
        figure.tight_layout()
        figure.savefig(out_dir / f"{activity_name}_grf_method_comparison_summary.png", dpi=160)
        plt.close(figure)
