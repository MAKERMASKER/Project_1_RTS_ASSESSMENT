"""fit_ideal_gains.py -- the "ideal" insole vertical-force gain per
subject/side/ACTIVITY, fit directly against force-plate ground truth.

Question
--------
The production gain (`vertical_force_scale` in
results/calibration/<ID>_insole_marker_foot_calibration.json) is fit ONCE per
subject/side from SLJ quiet single-leg stance only (calibration.py hardcodes
`activity_for("single_leg_jump")`), then applied unchanged to every activity.
Is that valid? If SLJ's stance gain is 0.78 and SLJ's ideal is 0.82, does that
same value still hold for lateral hop and skater hop?

This script fits, for every trial, the gain that would make the raw insole
vertical force best match the force plate -- the best any single scalar gain
could possibly do on that trial. Comparing those ideal gains ACROSS activities
answers whether one per-subject gain is defensible or whether gain is
activity-dependent.

**These gains are ground-truth-derived and are NOT field-deployable.** They
require a force plate, which is exactly what the framework pipeline exists to
avoid. They are a BENCHMARK / CEILING -- "how good could the gain possibly
be" -- and a test of gain transferability, never a deployable calibration.
Anything reprocessed with them is a benchmark configuration, not a validation
of the markerless pipeline's independent accuracy.

Method
------
Regression through the origin over contact frames, the same estimator used for
the OpenSimAD gain work earlier in this project:
    gain = sum(insole * plate) / sum(insole ** 2)
(A peak ratio is also reported for reference, but is NOT the fit -- single-
point matching was already established as misleading in this project.)

Axis note: this is vertical force (Fy) only. The ~90deg marker-lab vs. OpenCap
global frame rotation documented in CLAUDE.md affects Fx/Fz only and is
irrelevant here -- Fy is common to both frames.

Run from code/:
    ..\\opensimPy\\Scripts\\python.exe expt_temp/cross_activity_gain_check/fit_ideal_gains.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(CODE_ROOT))

from activities import activity_for  # noqa: E402
from calibration import InsoleCalibration  # noqa: E402
from config import DEFAULT  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from io_tables import read_insole, read_storage, read_trc  # noqa: E402
from subject import Subject  # noqa: E402

SUBJECT_IDS = ("AB03", "AB05", "AB07", "AB08", "AB10", "NTS01")
# TH deliberately excluded for now: its state detector is not yet finalised,
# so its insole<->OpenCap sync cannot be trusted for a gain fit.
ACTIVITIES = ("single_leg_jump", "lateral_hop", "skater_hop")
RESULTS = Path(__file__).resolve().parent / "results"


def _ideal_gain(trial, subject, activity, calibration):
    """One row per (trial, side) with a usable contact window."""
    trc = read_trc(trial.effective_opencap_trc())
    ik_all = read_storage(trial.effective_opencap_ik())
    insole_all = read_insole(trial.insole)
    time, insole, _ik, sync = activity.synchronize(trial, trc, ik_all, insole_all, subject.rules)
    # Honour the SAME quality gate the analysis stage applies
    # (outcomes.py's `if bool(data.get("quality_omit", False)): return None`).
    # A trial whose sync the activity itself distrusts cannot support a gain
    # fit either -- fitting one silently produces a confident-looking number
    # from mis-aligned data. Found for real: every NTS01 LH/SKH trial is
    # quality_omit, and fitting them yielded a wild 0.372 "ideal" gain against
    # that subject's 0.850 SLJ value.
    if bool(sync.get("quality_omit", False)):
        return []
    force_plate = read_storage(trial.force_mot)

    # The processed force MOT is always ground_force_1 = right, _2 = left
    # (io_tables.write_bilateral_grf), independent of activity.
    # trial.side is None for a bilateral activity (SkaterHop) -- fall back to
    # its own sync anchor side rather than silently mis-picking a plate.
    sides = [trial.side] if trial.side else [sync.get("sync_side", "R")]

    rows = []
    for side in sides:
        column = "right_force" if side == "R" else "left_force"
        plate_number = 1 if side == "R" else 2
        own_insole = insole[column].to_numpy(float)
        plate_raw = np.abs(force_plate[f"ground_force_{plate_number}_vy"].to_numpy(float))
        if not np.isfinite(own_insole).any() or own_insole.max() <= 0:
            continue

        shift = float(time[np.argmax(own_insole)] - force_plate.time.iloc[np.argmax(plate_raw)])
        plate = np.interp(time, force_plate.time + shift, plate_raw, left=np.nan, right=np.nan)

        contact = activity.contact(own_insole)
        use = contact & np.isfinite(plate) & (own_insole > 0)
        if use.sum() < 20:
            continue

        raw, truth = own_insole[use], plate[use]
        ideal = float(np.sum(raw * truth) / np.sum(raw ** 2))
        peak_ratio = float(np.max(truth) / np.max(raw)) if np.max(raw) > 0 else np.nan
        current = float(calibration[side]["vertical_force_scale"])

        def rmse(gain):
            return float(np.sqrt(np.mean((gain * raw - truth) ** 2)))

        rows.append({
            "Subject": subject.id, "Activity": activity.name, "Trial": trial.name,
            "Side": side, "Condition": "EXO" if "_EXO_" in trial.name else "NOEXO",
            "ContactFrames": int(use.sum()),
            "IdealGain": round(ideal, 4),
            "PeakRatioGain": round(peak_ratio, 4),
            "CurrentGain": round(current, 4),
            "IdealMinusCurrent": round(ideal - current, 4),
            "PctDiffVsCurrent": round(100 * (ideal - current) / current, 2) if current else np.nan,
            "RMSE_idealGain_N": round(rmse(ideal), 1),
            "RMSE_currentGain_N": round(rmse(current), 1),
            "PeakPlate_N": round(float(np.max(truth)), 1),
            "PeakRawInsole_N": round(float(np.max(raw)), 1),
            # Makes the cost of using the WRONG gain exactly computable:
            # for a through-origin fit the cross term vanishes, so
            #   RMSE(g)^2 = RMSE(ideal)^2 + (g - ideal)^2 * mean(raw^2)
            # i.e. added error (in quadrature) = |g - ideal| * RmsRawInsole_N.
            "RmsRawInsole_N": round(float(np.sqrt(np.mean(raw ** 2))), 1),
        })
    return rows


def main():
    discovery = TrialDiscovery(DEFAULT)
    calibration_obj = InsoleCalibration(DEFAULT)
    rows = []

    for subject_id in SUBJECT_IDS:
        subject = Subject(subject_id, DEFAULT)
        try:
            calibration = calibration_obj.load(subject)
        except FileNotFoundError:
            print(f"{subject_id}: no calibration -- skipping")
            continue
        for activity_name in ACTIVITIES:
            activity = activity_for(activity_name, DEFAULT)
            trials = [
                trial for trial in discovery.trials(subject, activity, "all", for_analysis=True)
                if trial.insole.is_file() and trial.force_mot.is_file()
                and trial.effective_opencap_trc() and Path(trial.effective_opencap_trc()).is_file()
                and trial.effective_opencap_ik() and Path(trial.effective_opencap_ik()).is_file()
            ]
            made = 0
            for trial in sorted(trials, key=lambda t: t.name):
                try:
                    new_rows = _ideal_gain(trial, subject, activity, calibration)
                    rows.extend(new_rows)
                    made += len(new_rows)
                except Exception as error:  # noqa: BLE001 -- report, keep going
                    print(f"  {trial.name}: FAILED ({type(error).__name__}: {error})")
            print(f"{subject_id}/{activity_name}: {made} fits from {len(trials)} trials")

    if not rows:
        print("Nothing fit.")
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "ideal_gains_per_trial.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} (trial, side) fits -> {out}")


if __name__ == "__main__":
    main()
