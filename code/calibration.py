"""InsoleCalibration: per-subject, per-side insole force gain + rigid COP offset.

Fit ONCE per subject (never per-activity -- see activities.py's module
docstring) and reused unchanged by FrameworkProcessor regardless of which
activity a trial belongs to.

Correction vs. the old reference pipeline (resources/_old_reference_code):
that code fit ONE vertical-force gain from combined bilateral (left+right)
stable-stance force and applied the same value to both feet -- physically
wrong, since the two insoles are independent sensors. Here, gain_L and gain_R
are fit independently, each from that side's own SLJ trials' own recorded
insole force -- never from a combined bilateral total. The rigid COP offset
was already correctly per-side in the old code and is kept as-is.

Scope note: this fits what the currently recorded data supports (no subject
has a dedicated quiet-stance calibration trial). That is a real limitation,
documented here and in resources/literature_notes.md, not something this
module works around by inventing data collection that doesn't exist yet.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from activities import activity_for
from config import DEFAULT
from discovery import TrialDiscovery
from insole_geometry import InsoleGeometry
from io_tables import read_insole, read_storage, read_trc
from opensim_env import lowpass


class InsoleCalibration:
    # Trigger value for subject_rules force_scale_mode[side] that means "fit
    # from this side's own stable-force samples." Anything else that isn't
    # "fixed" means no scaling (gain = 1) -- matches the old code's fallback.
    AUTO_MODE = "body_weight_from_stable_side_force"
    ERODE_GUARD_S = 0.03

    # Genuine single-leg quiet-stance detection (see _stance_window): a
    # candidate sample requires the OPPOSITE foot unloaded (proves it's real
    # single-leg support, not bilateral standing) and the own foot
    # meaningfully weight-bearing with a low instantaneous gradient. Not
    # every trial contains a stretch this good for long enough -- see
    # _stance_window's docstring for why duration + CV are the real quality
    # gate, and why a trial with none is skipped rather than forced.
    STANCE_LOWPASS_HZ = 6.0
    STANCE_OPPOSITE_ZERO_FRACTION = 0.05
    STANCE_WEIGHT_BEARING_FRACTION = 0.5
    STANCE_GRADIENT_MAX_N_PER_S = 200.0
    STANCE_MIN_DURATION_S = 0.4
    STANCE_MAX_CV = 0.06
    # Genuine quiet single-leg stance means ~body weight on that one foot;
    # a flat, low-CV window that's actually 60% over/under body weight is a
    # transient (e.g. a brief mid-landing hold), not stance -- caught for
    # real on AB08_NOEXO_SLJ_R_8 (0.43s, CV 0.4%, but median 1.62x weight).
    STANCE_MIN_WEIGHT_FRACTION = 0.7
    STANCE_MAX_WEIGHT_FRACTION = 1.3

    def __init__(self, config=DEFAULT):
        self.config = config
        self.discovery = TrialDiscovery(config)
        self.activity = activity_for("single_leg_jump", config)

    def path(self, subject) -> Path:
        return self.config.calibration_root / f"{subject.id}_insole_marker_foot_calibration.json"

    def exists(self, subject) -> bool:
        return self.path(subject).is_file()

    def load(self, subject) -> dict:
        calibration_path = self.path(subject)
        if not calibration_path.is_file():
            raise FileNotFoundError(
                f"Missing calibration for {subject.id}: run InsoleCalibration.fit() first ({calibration_path})"
            )
        return json.loads(calibration_path.read_text())

    # ---- fitting -----------------------------------------------------

    def fit(self, subject) -> Path:
        geometry = InsoleGeometry.for_subject(subject)
        rules = subject.rules

        gain_info = {side: self._force_gain(subject, side, rules) for side in ("L", "R")}

        records = []
        for condition in self.config.conditions:
            for trial in self.discovery.trials(subject, self.activity, condition, for_analysis=True):
                record = self._trial_delta(trial, geometry)
                if record:
                    records.append(record)

        calibration = {
            side: {
                **geometry.shape,
                "vertical_force_scale": gain_info[side][0],
                "vertical_force_scale_source": gain_info[side][1],
                "rigid_offset_local_m": [0.0, 0.0, 0.0],
                "cop_mapping": "measured_geometry_plus_rigid_translation",
            }
            for side in ("L", "R")
        }
        for side in ("L", "R"):
            values = [record["delta"] for record in records if record["side"] == side]
            if values:
                calibration[side]["rigid_offset_local_m"] = [
                    *np.median(np.vstack(values), axis=0).tolist(), 0.0
                ]

        calibration["trial_offsets"] = {}
        for record in records:
            others = [r["delta"] for r in records if r is not record and r["side"] == record["side"]]
            offset = (
                [*np.median(np.vstack(others), axis=0).tolist(), 0.0]
                if others else calibration[record["side"]]["rigid_offset_local_m"]
            )
            calibration["trial_offsets"][record["name"]] = {
                "rigid_offset_local_m": offset,
                "method": "same-side leave-one-trial-out",
            }
        calibration["version"] = "per_side_v1"
        calibration["geometry_source"] = "measured insole dimensions"

        self.config.calibration_root.mkdir(parents=True, exist_ok=True)
        calibration_path = self.path(subject)
        calibration_path.write_text(json.dumps(calibration, indent=2))

        self._write_audit(subject, records, gain_info)
        return calibration_path

    def _write_audit(self, subject, records, gain_info):
        out = self.config.results_root / subject.id / "analysis" / "insole_calibration"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {
                "Trial": r["name"], "Side": r["side"], "Samples": r["samples"],
                "MedianApDelta_m": np.median(r["delta"][:, 0]),
                "MedianMlDelta_m": np.median(r["delta"][:, 1]),
                "PeakSyncShift_s": r["shift"],
            }
            for r in records
        ]).to_csv(out / "rigid_offset_calibration_audit.csv", index=False)
        pd.DataFrame([
            {
                "Subject": subject.id, "Side": side,
                "VerticalForceScale": gain, "VerticalForceScaleSource": source,
                "AcceptedStanceTrials": count,
            }
            for side, (gain, source, _weight, _median, count, _reports) in gain_info.items()
        ]).to_csv(out / "vertical_force_scale_audit.csv", index=False)
        pd.DataFrame([
            {"Subject": subject.id, "Side": side, **report}
            for side, (*_rest, reports) in gain_info.items()
            for report in reports
        ]).to_csv(out / "stance_window_selection_audit.csv", index=False)

    def _force_gain(self, subject, side: str, rules: dict):
        """gain, source, body_weight_n, median_of_accepted_n, accepted_trial_count,
        per-trial stance-window reports -- for one side only."""
        mode = rules["force_scale_mode"][side]
        weight = subject.mass_kg * self.config.gravity_mps2 if subject.mass_kg else None

        if mode == "fixed":
            return rules["force_scale"][side], "fixed_value", weight, np.nan, 0, []
        if mode != self.AUTO_MODE:
            return 1.0, "no_force_gain", weight, np.nan, 0, []
        if weight is None:
            raise ValueError(f"{subject.id} is missing Mass_kg metadata; cannot auto-fit force gain")

        own_column = "left_force" if side == "L" else "right_force"
        opposite_column = "right_force" if side == "L" else "left_force"

        accepted = []
        reports = []
        for condition in self.config.conditions:
            for trial in self.discovery.trials(subject, self.activity, condition):
                if trial.side != side or not trial.insole.is_file():
                    continue
                insole = read_insole(trial.insole)
                dt = float(np.median(np.diff(insole.time)))
                window = self._stance_window(
                    insole[own_column].to_numpy(float),
                    insole[opposite_column].to_numpy(float),
                    dt, weight,
                )
                if window is None:
                    reports.append({"Trial": trial.name, "Used": False, "Reason": "no_candidate_window"})
                    continue
                reports.append({
                    "Trial": trial.name, "Used": window["accepted"],
                    "Reason": "ok" if window["accepted"] else "candidate_too_variable_or_implausible",
                    "DurationS": window["duration"], "CV": window["cv"], "MedianN": window["median"],
                    "WeightFraction": window["weight_fraction"],
                })
                if window["accepted"]:
                    accepted.append(window["median"])

        if not accepted:
            raise ValueError(
                f"No {side}-side SLJ trial for {subject.id} contains a qualifying quiet "
                f"single-leg-stance window (>= {self.STANCE_MIN_DURATION_S}s, "
                f"CV <= {self.STANCE_MAX_CV:.0%}) -- see "
                f"results/{subject.id}/analysis/insole_calibration/stance_window_selection_audit.csv "
                "(once run with a relaxed threshold) or validation/sensor_insole/RAW_visualization; "
                "consider a 'fixed' override in subject_rules or a dedicated stance calibration trial."
            )
        median = float(np.median(accepted))
        source = "body_weight_over_median_of_accepted_trial_stance_medians"
        return weight / median, source, weight, median, len(accepted), reports

    @classmethod
    def _stance_window(cls, own_force: np.ndarray, opposite_force: np.ndarray, dt: float, weight_n: float):
        """The best genuine single-leg quiet-stance window in one trial, or
        None if this trial has no candidate window at all.

        A sample only qualifies as single-leg support if the OPPOSITE foot's
        (low-pass filtered) force is near zero -- proof this is real
        single-leg stance, not bilateral standing, which a pointwise
        force/gradient test alone cannot distinguish. Own-side force must
        also be meaningfully weight-bearing with a low instantaneous
        gradient. But a slow, continuous sway can pass all three of those
        pointwise tests at scattered moments without ever being a genuine
        plateau -- so the real quality gate is on the resulting CONTIGUOUS
        RUN: it must last >= STANCE_MIN_DURATION_S, have a coefficient of
        variation <= STANCE_MAX_CV (see AB05_EXO_SLJ_R_1 vs. R_4 in
        validation/sensor_insole/RAW_visualization -- R_1's post-swing
        region keeps satisfying the pointwise tests while visibly wobbling,
        R_4's doesn't wobble), AND read close to body weight (a flat-but-
        off-weight run is a transient hold mid-landing, not quiet stance --
        caught for real on AB08_NOEXO_SLJ_R_8: 0.43s, CV 0.4%, but median
        1.62x body weight). Returns the longest run that clears the
        duration bar, tagged accepted=True only if it also clears the CV
        and weight-plausibility bars, so a trial can be reported (and
        audited) even when rejected rather than silently disappearing.
        """
        time = np.arange(len(own_force)) * dt
        own_filt = lowpass(own_force, time, cls.STANCE_LOWPASS_HZ)
        opposite_filt = lowpass(opposite_force, time, cls.STANCE_LOWPASS_HZ)
        gradient = np.abs(np.gradient(own_filt, dt))

        candidate = (
            (opposite_filt <= cls.STANCE_OPPOSITE_ZERO_FRACTION * weight_n)
            & (own_filt >= cls.STANCE_WEIGHT_BEARING_FRACTION * weight_n)
            & (gradient <= cls.STANCE_GRADIENT_MAX_N_PER_S)
            & np.isfinite(own_filt) & np.isfinite(opposite_filt)
        )

        runs = []
        for start, end in cls._contiguous_runs(candidate):
            duration = (end - start) * dt
            if duration < cls.STANCE_MIN_DURATION_S:
                continue
            segment = own_filt[start:end]
            mean = float(np.mean(segment))
            cv = float(np.std(segment) / mean) if mean else np.inf
            weight_fraction = mean / weight_n if weight_n else np.inf
            plausible = cls.STANCE_MIN_WEIGHT_FRACTION <= weight_fraction <= cls.STANCE_MAX_WEIGHT_FRACTION
            runs.append({
                "duration": duration, "cv": cv, "median": float(np.median(segment)),
                "weight_fraction": weight_fraction,
                "accepted": cv <= cls.STANCE_MAX_CV and plausible,
            })
        if not runs:
            return None
        passing = [run for run in runs if run["accepted"]]
        pool = passing if passing else runs
        return max(pool, key=lambda run: run["duration"])

    @staticmethod
    def _contiguous_runs(mask: np.ndarray):
        """[start, end) index pairs for each maximal run of True in mask."""
        runs = []
        start = None
        for index, value in enumerate(np.append(np.asarray(mask, bool), False)):
            if value and start is None:
                start = index
            elif not value and start is not None:
                runs.append((start, index))
                start = None
        return runs

    # ---- rigid COP offset (unchanged from the old code: already per-side) --

    @staticmethod
    def _erode(mask, guard: int):
        mask = np.asarray(mask, bool)
        out = np.zeros_like(mask)
        for index in range(guard, len(mask) - guard):
            out[index] = mask[index - guard:index + guard + 1].all()
        return out

    def _trial_delta(self, trial, geometry: InsoleGeometry):
        if not all(path.is_file() for path in (trial.insole, trial.force_mot, trial.marker_trc)):
            return None
        insole = read_insole(trial.insole)
        force_plate = read_storage(trial.force_mot)
        trc = read_trc(trial.marker_trc)
        word = "right" if trial.side == "R" else "left"
        plate = 1 if trial.side == "R" else 2

        force = insole[f"{word}_force"].to_numpy(float)
        vertical = np.abs(force_plate[f"ground_force_{plate}_vy"].to_numpy(float))
        cop = force_plate[[f"ground_force_{plate}_p{axis}" for axis in "xyz"]].to_numpy(float)
        if np.nanmedian(np.abs(cop)) > 2:
            cop /= 1000  # mm -> m

        shift = float(
            force_plate.time.iloc[int(np.argmax(vertical))]
            - insole.time.iloc[int(np.argmax(np.convolve(force, np.ones(7) / 7, mode="same")))]
        )
        native = geometry.native_cop(insole[f"{word}_cop_x"], insole[f"{word}_cop_y"], trial.side)
        aligned = np.column_stack([
            np.interp(force_plate.time, insole.time + shift, native[:, axis], left=np.nan, right=np.nan)
            for axis in range(3)
        ])
        active = np.interp(force_plate.time, insole.time + shift, force, left=0, right=0)
        local = InsoleGeometry.to_local(trc, force_plate.time.to_numpy(), cop, trial.side)

        use = (
            (active >= self.config.contact_force_n)
            & (vertical >= self.config.contact_force_n)
            & np.isfinite(np.c_[local[:, :2], aligned[:, :2]]).all(axis=1)
        )
        dt = float(np.median(np.diff(force_plate.time)))
        use = self._erode(use, int(np.ceil(self.ERODE_GUARD_S / dt)))
        delta = local[use, :2] - aligned[use, :2]
        if not len(delta):
            return None
        return {"name": trial.name, "side": trial.side, "delta": delta, "samples": len(delta), "shift": shift}
