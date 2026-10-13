"""extract_phases.py -- SLJ joint kinetics split by PHASE and CONTRACTION
MODE, for every subject that has EXO isokinetic data.

Generalised from expt_temp/AB03_FullData_Iso_association/extract_phases.py,
which established the decomposition on one subject. See that README for why an
extensor-vs-flexor split of the knee moment does not work: every joint carries
an extensor moment in BOTH phases of a single-leg jump, so the flexor bin
collects only noise (0-85 Nm against a 165-336 Nm extensor peak) and yields
LSIs with 281-654 point confidence intervals.

The usable split is phase and power sign:
  * PUSH-OFF -- extensor moment, power POSITIVE (concentric, generating)
  * LANDING  -- extensor moment, power NEGATIVE (eccentric, absorbing)

Sign convention: OpenSim reports generalized forces on the FLEXION coordinate,
so a negative moment is an EXTENSOR moment. Verified on real data rather than
assumed (knee angle falls 68.8 -> 4.1 deg through push-off while the moment
stays negative and power is positive: concentric quadriceps).

Alignment and the landing window replicate `OutcomeExtractor.analyze_trial()`,
so these share a basis with the production trial_summary.

Run from code/:
    ..\\opensimPy\\Scripts\\python.exe expt_temp/EXO_device_signature/extract_phases.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(CODE_ROOT))

from activities import activity_for  # noqa: E402
from config import DEFAULT  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from io_tables import read_storage, segments  # noqa: E402
from subject import Subject  # noqa: E402

# Subjects with BOTH movement data and EXO isokinetic data. AB02/AB04/AB09
# have isokinetic but no data folder at all; AB10/NTS01 have movement but no
# isokinetic row.
SUBJECT_IDS = ("AB03", "AB05", "AB07", "AB08")
ACTIVITY_NAME = "single_leg_jump"
RESULTS = Path(__file__).resolve().parent / "results"
JOINTS = ("Hip", "Knee", "Ankle")


def _interp(source_time, values, target_time):
    return np.interp(target_time, source_time, values, left=np.nan, right=np.nan)


def _phase_metrics(label, phase, moment, angle, support, time, window):
    out = {}
    window_time = time[window]
    velocity = np.gradient(np.deg2rad(angle), time, axis=0)
    power = moment * velocity
    for index, joint in enumerate(JOINTS):
        extensor = np.maximum(-moment[:, index], 0.0)[window]
        joint_power = power[window, index]
        prefix = f"{label}_{phase}_{joint}"
        out[f"{prefix}ExtPeak_Nm"] = round(float(np.nanmax(extensor)), 2)
        out[f"{prefix}ExtImpulse_Nms"] = round(float(np.trapezoid(extensor, window_time)), 3)
        out[f"{prefix}ConcWork_J"] = round(
            float(np.trapezoid(np.maximum(joint_power, 0.0), window_time)), 2)
        out[f"{prefix}EccWork_J"] = round(
            float(np.trapezoid(np.abs(np.minimum(joint_power, 0.0)), window_time)), 2)
        out[f"{prefix}MeanVel_degs"] = round(
            float(np.nanmean(np.abs(np.gradient(angle[:, index], time)[window]))), 1)
    out[f"{label}_{phase}_SupportPeak_Nm"] = round(float(np.nanmax(support[window])), 2)
    out[f"{label}_{phase}_SupportImpulse_Nms"] = round(
        float(np.trapezoid(support[window], window_time)), 3)
    return out


def _trial_row(subject, trial):
    package = trial.output / "trial_data.npz"
    framework_id = trial.output / f"{trial.name}_opencap_insole_id.sto"
    marker_root = DEFAULT.results_root / subject.id / "marker_based" / trial.condition
    marker_ik_path = marker_root / "ik" / f"{trial.name}_ik.mot"
    marker_id_path = marker_root / "id" / f"{trial.name}_inverse_dynamics.sto"
    if not all(Path(p).is_file() for p in
               (package, framework_id, marker_ik_path, marker_id_path, trial.force_mot)):
        return None

    data = np.load(package, allow_pickle=True)
    if bool(data.get("quality_omit", False)):
        return None
    time = data["time"]
    contact = data["contact"].astype(bool)
    grf = data["grf_N"]

    framework_ik = pd.DataFrame(data["ik_values"], columns=[str(n) for n in data["ik_names"]])
    framework_id_table = read_storage(framework_id)
    marker_id_table = read_storage(marker_id_path)
    marker_ik_table = read_storage(marker_ik_path)
    force_plate = read_storage(trial.force_mot)

    side = trial.side
    plate = 1 if side == "R" else 2
    marker_vertical = np.abs(force_plate[f"ground_force_{plate}_vy"].to_numpy(float))
    shift = float(time[np.argmax(data["force"])] - force_plate.time.iloc[np.argmax(marker_vertical)])

    suffix = side.lower()
    angle_names = [f"hip_flexion_{suffix}", f"knee_angle_{suffix}", f"ankle_angle_{suffix}"]
    moment_names = [f"{a}_moment" for a in angle_names]
    framework_angle = np.column_stack([framework_ik[a] for a in angle_names])
    marker_angle = np.column_stack([
        _interp(marker_ik_table.time + shift, marker_ik_table[a], time) for a in angle_names])
    framework_moment = np.column_stack([
        np.interp(time, framework_id_table.time, framework_id_table[m]) for m in moment_names])
    marker_moment = np.column_stack([
        _interp(marker_id_table.time + shift, marker_id_table[m], time) for m in moment_names])

    contacts = segments(contact)
    peak_index = int(np.argmax(grf[:, 1]))
    landing_at = [i for i, (s, e) in enumerate(contacts) if s <= peak_index <= e]
    if not landing_at:
        return None
    landing_index = landing_at[0]

    row = {"Subject": subject.id, "Trial": trial.name,
           "Condition": trial.condition, "Side": side,
           "HasPushOff": landing_index > 0}
    dt = float(np.median(np.diff(time)))
    phases = {"Landing": contacts[landing_index]}
    if landing_index > 0:
        phases["PushOff"] = contacts[landing_index - 1]
    for phase, (start, end) in phases.items():
        window = slice(start, end + 1)
        row[f"{phase}Duration_s"] = round((end - start + 1) * dt, 4)
        for label, moment, angle in (("Framework", framework_moment, framework_angle),
                                     ("Marker", marker_moment, marker_angle)):
            support = -np.sum(moment, axis=1)
            row.update(_phase_metrics(label, phase, moment, angle, support, time, window))
    return row


def main():
    activity = activity_for(ACTIVITY_NAME, DEFAULT)
    discovery = TrialDiscovery(DEFAULT)
    rows = []
    for subject_id in SUBJECT_IDS:
        subject = Subject(subject_id, DEFAULT)
        made = 0
        for condition in DEFAULT.conditions:
            for trial in discovery.trials(subject, activity, condition, for_analysis=True):
                try:
                    row = _trial_row(subject, trial)
                except Exception as error:  # noqa: BLE001
                    print(f"  {trial.name}: FAILED ({type(error).__name__}: {error})")
                    continue
                if row is None:
                    continue
                rows.append(row)
                made += 1
        print(f"{subject_id}: {made} trials")

    table = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "phase_kinetics.csv", index=False)
    print(f"\n{len(table)} trials -> {RESULTS / 'phase_kinetics.csv'}")
    print(f"with a captured push-off contact: {int(table.HasPushOff.sum())}/{len(table)}")
    print()
    print(table.groupby(["Subject", "Condition", "Side"]).size().to_string())


if __name__ == "__main__":
    main()
