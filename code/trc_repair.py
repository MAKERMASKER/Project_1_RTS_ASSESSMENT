"""TRC-repair QC: marker-trajectory quality control for OpenCap TRC files.

Ported from the in-progress prototype at
G:\\Shared drives\\SuhrudLib\\PILOT_OOP_BASE\\temp\\AB08_SLJ_moment_investigation\\trc_repair_gui\\.

The workflow, three explicit steps -- see cli.py's `repair-filter` /
`repair-validate` / `repair-promote` commands:
  1. AutomaticTrcFilter: batch 10 Hz zero-phase low-pass filtering of a whole
     trial's markers (skips a trial if it has any missing sample rather than
     guessing), with a per-marker quality report flagging unusually large
     change (median + 5*MAD) for human review -- a review aid, not an
     automatic pass/fail. Writes a candidate filtered TRC; nothing production
     reads yet.
  2. TrcRepairValidator: matched original-vs-repaired validation. Reruns
     OpenCap's own IK setup (via opencap_ik.OpenCapIkRunner -- the same class
     framework_processor.py uses to regenerate *production* IK for a promoted
     repair, so validation and production use the identical recipe) and the
     framework pipeline on both the untouched and repaired TRC, then compares
     hip/knee/ankle angle and moment against the marker/ground-truth pipeline
     -- so any change is attributable only to the repair, never to a
     different pipeline run.
  3. RepairPromoter: the ONLY step that makes a repair take effect. Copies
     the candidate TRC to `PROCESSED/OPENCAP_REPAIRED/<name>.trc` and adds
     the trial name to subject_rules/<ID>.json's `repaired_trials` list --
     the human confirmation gate `discovery.py`/`Trial.effective_opencap_trc`
     check before a repair is ever used. Only *after* this runs does the next
     `framework` command see the repair and rerun IK from it automatically
     (framework_processor.py's `process()`) -- IK is never rerun before a
     human has explicitly promoted the repair.

NOT ported: the interactive matplotlib GUI (trc_repair_gui.py's RepairGUI)
for manual per-marker, per-interval review. That remains future work; when
it's ported it will reuse TrcRepairValidator for its own "Run validation"
button, exactly like the automatic path above does, and RepairPromoter for
its own "Promote" action.

Production gate (unchanged from the prototype's README): a repair is
eligible for real use only after (1) every changed interval is reviewed,
(2) the audit is retained, (3) IK marker error and lower-limb kinematics
improve without creating discontinuities, (4) moment changes are
interpreted via this matched comparison, (5) improvements are consistent
across more than one accepted trial. This module never promotes a repair on
its own -- `repair-promote` is a separate, explicit CLI invocation a human
runs only after reviewing steps 1-2's output.
"""
from dataclasses import replace
from pathlib import Path
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from config import DEFAULT
from framework_processor import FrameworkProcessor
from io_tables import read_storage, read_trc, write_trc
from subject import Subject
from opencap_ik import OpenCapIkRunner


def _rms(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values ** 2))) if len(values) else np.nan


def _at(table: pd.DataFrame, name: str, time):
    return np.interp(time, table.time, table[name], left=np.nan, right=np.nan)


class AutomaticTrcFilter:
    """Whole-trial 10 Hz zero-phase low-pass filter + per-marker quality report."""

    CUTOFF_HZ = 10.0

    def filtered_xyz(self, trc: dict) -> np.ndarray:
        if not np.isfinite(trc["xyz"]).all():
            raise ValueError("TRC contains gaps; automatic filtering was not applied.")
        sample_rate = 1.0 / float(np.median(np.diff(trc["time"])))
        if self.CUTOFF_HZ >= sample_rate / 2:
            raise ValueError("Cutoff must be below the TRC's Nyquist frequency.")
        sections = butter(4, self.CUTOFF_HZ, fs=sample_rate, output="sos")
        return sosfiltfilt(sections, trc["xyz"], axis=0)

    def marker_quality(self, trc: dict, filtered: np.ndarray) -> pd.DataFrame:
        rows = []
        for index, name in enumerate(trc["marker_names"]):
            raw = trc["xyz"][:, index, :]
            clean = filtered[:, index, :]
            change = np.linalg.norm(clean - raw, axis=1) * 1000.0
            raw_roughness = _rms(np.diff(raw, n=2, axis=0)) * 1000.0
            clean_roughness = _rms(np.diff(clean, n=2, axis=0)) * 1000.0
            reduction = 100.0 * (1.0 - clean_roughness / raw_roughness) if raw_roughness else 0.0
            rows.append({
                "Marker": name,
                "RMSChange_mm": _rms(change),
                "MaxChange_mm": float(np.max(change)),
                "RawRoughness_mm": raw_roughness,
                "FilteredRoughness_mm": clean_roughness,
                "RoughnessReduction_pct": reduction,
            })
        table = pd.DataFrame(rows)
        median = float(table.RMSChange_mm.median())
        mad = float(np.median(np.abs(table.RMSChange_mm - median)))
        limit = median + 5.0 * 1.4826 * mad
        table["ReviewUnusualChange"] = table.RMSChange_mm > limit if mad > 0 else False
        return table

    def process_trial(self, trial, out_dir: Path) -> dict:
        """Filter one trial's OpenCap TRC; write the filtered TRC + quality report."""
        out_dir.mkdir(parents=True, exist_ok=True)
        source = out_dir / "source_original.trc"
        filtered_path = out_dir / "filtered_10hz.trc"
        shutil.copy2(trial.opencap_trc, source)

        trc = read_trc(source)
        filtered = self.filtered_xyz(trc)
        quality = self.marker_quality(trc, filtered)
        quality.to_csv(out_dir / "marker_quality.csv", index=False)
        write_trc(source, filtered_path, filtered)
        return {
            "source_trc": source, "filtered_trc": filtered_path, "quality": quality,
            "markers_flagged_for_review": int(quality["ReviewUnusualChange"].sum()),
        }


class TrcRepairValidator:
    """Matched original-vs-repaired IK + framework ID comparison against marker ground truth."""

    def __init__(self, config=DEFAULT):
        self.config = config
        self.framework_processor = FrameworkProcessor(config)
        self.ik_runner = OpenCapIkRunner(config)

    def validate(self, subject, activity, trial, repaired_trc_path, out_dir: Path = None, side: str = None) -> pd.DataFrame:
        """Compare a candidate repair against the untouched TRC and the marker
        ground truth. Writes CSVs + a comparison plot; returns the per-joint
        summary. Promoting the repair is a separate, human decision.

        `side`: which leg's hip/knee/ankle angle+moment to validate against.
        Defaults to trial.side (every unilateral activity: SLJ, LH). For a
        bilateral activity with no committed side (trial.side is None, e.g.
        SkaterHop), pass the activity's fixed sync side explicitly (SKH
        always syncs/validates on "R" -- see SkaterHop's sync_side="R" in
        activities.py) -- this cannot be inferred from the trial alone."""
        side = side or trial.side
        if side is None:
            raise ValueError(
                f"{trial.name}: trial.side is None (bilateral activity) and no "
                f"explicit side was passed to validate() -- cannot pick which "
                f"leg's angle/moment to compare."
            )
        out_dir = Path(out_dir) if out_dir else trial.output / "trc_repair_validation"
        local_original_dir = out_dir / "local_original"
        local_repaired_dir = out_dir / "local_repaired"
        local_original_dir.mkdir(parents=True, exist_ok=True)
        local_repaired_dir.mkdir(parents=True, exist_ok=True)

        original_ik_path = local_original_dir / f"{trial.name}_local_original_ik.mot"
        self.ik_runner.run(trial.opencap_trc, trial.opencap_model, original_ik_path, local_original_dir)
        repaired_ik_path = local_repaired_dir / f"{trial.name}_local_repaired_ik.mot"
        self.ik_runner.run(repaired_trc_path, trial.opencap_model, repaired_ik_path, local_repaired_dir)

        original_trial = replace(
            trial, opencap_trc=trial.opencap_trc, opencap_ik=original_ik_path, output=local_original_dir
        )
        repaired_trial = replace(
            trial, opencap_trc=Path(repaired_trc_path), opencap_ik=repaired_ik_path, output=local_repaired_dir
        )
        self.framework_processor.process(subject, activity, original_trial)
        self.framework_processor.process(subject, activity, repaired_trial)

        repaired_data = np.load(local_repaired_dir / "trial_data.npz", allow_pickle=True)
        time = repaired_data["time"].astype(float)
        side = side.lower()
        angle_names = [f"hip_flexion_{side}", f"knee_angle_{side}", f"ankle_angle_{side}"]
        moment_names = [f"{name}_moment" for name in angle_names]

        baseline_ik = read_storage(original_ik_path)
        repaired_ik = read_storage(repaired_ik_path)
        marker_root = self.config.results_root / subject.id / "marker_based" / trial.condition
        marker_ik = read_storage(marker_root / "ik" / f"{trial.name}_ik.mot")
        baseline_id = read_storage(local_original_dir / f"{trial.name}_opencap_insole_id.sto")
        repaired_id = read_storage(local_repaired_dir / f"{trial.name}_opencap_insole_id.sto")
        marker_id = read_storage(marker_root / "id" / f"{trial.name}_inverse_dynamics.sto")

        force_plate = read_storage(trial.force_mot)
        plate = 1 if side == "r" else 2
        vertical = np.abs(force_plate[f"ground_force_{plate}_vy"].to_numpy())
        shift = float(time[np.argmax(repaired_data["force"])] - force_plate.time.iloc[np.argmax(vertical)])
        marker_time = time - shift
        contact = repaired_data["contact"].astype(bool)

        coordinate_names = [name for name in baseline_ik.columns if name != "time" and name in repaired_ik.columns]
        repair_effect = pd.DataFrame([
            {
                "Coordinate": name,
                "RepairChangeRMS": _rms(delta := _at(repaired_ik, name, baseline_ik.time.to_numpy(float)) - baseline_ik[name].to_numpy(float)),
                "RepairChangeMaxAbs": float(np.nanmax(np.abs(delta))),
            }
            for name in coordinate_names
        ])
        repair_effect.to_csv(out_dir / "repair_effect_on_ik.csv", index=False)

        joint_rows = []
        figure, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True)
        x = time - time[np.argmax(repaired_data["force"])]
        for column, joint in enumerate(("Hip", "Knee", "Ankle")):
            baseline_angle = _at(baseline_ik, angle_names[column], time)
            repaired_angle = _at(repaired_ik, angle_names[column], time)
            marker_angle = _at(marker_ik, angle_names[column], marker_time)
            baseline_moment = _at(baseline_id, moment_names[column], time)
            repaired_moment = _at(repaired_id, moment_names[column], time)
            marker_moment = _at(marker_id, moment_names[column], marker_time)

            baseline_angle_rms = _rms(baseline_angle - marker_angle)
            repaired_angle_rms = _rms(repaired_angle - marker_angle)
            baseline_moment_rms = _rms((baseline_moment - marker_moment)[contact])
            repaired_moment_rms = _rms((repaired_moment - marker_moment)[contact])
            joint_rows.append({
                "Joint": joint,
                "BaselineAngleRmsToMarker_deg": baseline_angle_rms,
                "RepairedAngleRmsToMarker_deg": repaired_angle_rms,
                "AngleRmsImprovement_deg": baseline_angle_rms - repaired_angle_rms,
                "BaselineMomentRmsToMarker_Nm": baseline_moment_rms,
                "RepairedMomentRmsToMarker_Nm": repaired_moment_rms,
                "MomentRmsImprovement_Nm": baseline_moment_rms - repaired_moment_rms,
            })

            axes[0, column].plot(x, marker_angle, "k", label="Marker")
            axes[0, column].plot(x, baseline_angle, color="tab:red", alpha=0.65, label="Original TRC")
            axes[0, column].plot(x, repaired_angle, color="tab:blue", label="Repaired TRC")
            axes[0, column].set_title(f"{joint} angle")
            axes[1, column].plot(x, marker_moment, "k")
            axes[1, column].plot(x, baseline_moment, color="tab:red", alpha=0.65)
            axes[1, column].plot(x, repaired_moment, color="tab:blue")
            axes[1, column].set_title(f"{joint} moment")
            axes[2, column].plot(x, repaired_moment - marker_moment, color="tab:blue", label="Repaired-marker")
            axes[2, column].plot(x, baseline_moment - marker_moment, color="tab:red", alpha=0.65, label="Baseline-marker")
            axes[2, column].set_title(f"{joint} moment error")
            for axis in axes[:, column]:
                axis.grid(True)

        axes[0, 0].legend()
        figure.suptitle(f"{trial.name} repaired TRC validation")
        figure.tight_layout()
        figure.savefig(out_dir / "repaired_vs_baseline_vs_marker.png", dpi=180)
        plt.close(figure)

        summary = pd.DataFrame(joint_rows)
        summary.to_csv(out_dir / "repair_validation_summary.csv", index=False)
        return summary


class RepairPromoter:
    """The human confirmation gate. The only thing that makes a repair take
    effect: copies a candidate TRC into PROCESSED/OPENCAP_REPAIRED/ and adds
    the trial to subject_rules/<ID>.json's repaired_trials list. Nothing
    before this point (AutomaticTrcFilter, TrcRepairValidator) changes what
    production processing reads -- only this does, and only when called
    explicitly (cli.py's `repair-promote`)."""

    def __init__(self, config=DEFAULT):
        self.config = config

    def promote(self, subject_id: str, trial, filtered_trc_path) -> Path:
        rules_path = self.config.subject_rules_root / f"{subject_id}.json"
        rules = json.loads(rules_path.read_text()) if rules_path.is_file() else {}
        repaired = set(rules.get("repaired_trials", []))
        repaired.add(trial.name)
        rules["repaired_trials"] = sorted(repaired)
        rules_path.write_text(json.dumps(rules, indent=4))

        # Re-read: a fresh Subject picks up the file just written above, so
        # opencap_aliases (if any) resolves correctly for the destination name.
        subject = Subject(subject_id, self.config)
        opencap_name = subject.rules["opencap_aliases"].get(trial.name, trial.name)
        destination = subject.data_folder / "PROCESSED" / "OPENCAP_REPAIRED" / f"{opencap_name}.trc"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(filtered_trc_path, destination)
        return destination
