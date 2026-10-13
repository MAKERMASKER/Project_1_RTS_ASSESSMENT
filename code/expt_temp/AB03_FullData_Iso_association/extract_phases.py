"""extract_phases.py -- AB03 SLJ joint kinetics split by PHASE and
CONTRACTION MODE, which is the decomposition the isokinetic reference can
actually be compared against.

Why not extensor-vs-flexor moment
---------------------------------
Measured on AB03: every joint (hip, knee, ankle) carries a NEGATIVE
(extensor) moment in BOTH the push-off and landing contacts. The lower limb
is extensor-dominant throughout a single-leg jump; only the POWER sign flips.
Splitting the knee moment into "extensor" and "flexor" bins therefore puts
0-85 Nm of noise in the flexor bin (against a 165-336 Nm extensor peak) and
produces LSIs with 281-654 point confidence intervals -- see README.

The meaningful split is:
  * PUSH-OFF  -- knee/hip/ankle extend, extensor moment, power POSITIVE
                 (concentric, generating)
  * LANDING   -- knee/hip flex, extensor moment still, power NEGATIVE
                 (eccentric, absorbing)

Isokinetic here is `Isokinetic Con/Con`, i.e. CONCENTRIC for both extension
and flexion. So push-off is mode-matched to the reference and landing is not;
both are extracted, and the mismatch is recorded rather than hidden.

Sign convention (verified on real data, not assumed): OpenSim reports
generalized forces on the FLEXION coordinate, so a negative moment is an
EXTENSOR moment. Confirmed because `knee_angle_r` falls 68.8 -> 4.1 deg
through push-off (the knee straightening) while the moment stays negative and
power is positive -- concentric quadriceps.

Support moment is Winter's whole-limb extensor sum, `-(hip + knee + ankle)`,
matching `outcomes.py`. It is an extensor quantity by construction.

Run from code/:
    ..\\opensimPy\\Scripts\\python.exe expt_temp/AB03_FullData_Iso_association/extract_phases.py
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

SUBJECT_ID = "AB03"
ACTIVITY_NAME = "single_leg_jump"
RESULTS = Path(__file__).resolve().parent / "results"
JOINTS = ("Hip", "Knee", "Ankle")


def _interp(source_time, values, target_time):
    return np.interp(target_time, source_time, values, left=np.nan, right=np.nan)


def _phase_metrics(label, phase, moment, angle, support, time, window):
    """Per-joint extensor peak/impulse and concentric/eccentric work."""
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

    # LANDING is the contact containing the peak vertical GRF (same definition
    # outcomes.py uses). PUSH-OFF is the contact immediately before it -- the
    # take-off that produced the flight, which sits inside SLJ's +/-0.75 s
    # window. Not every trial has one captured.
    contacts = segments(contact)
    peak_index = int(np.argmax(grf[:, 1]))
    landing_at = [i for i, (s, e) in enumerate(contacts) if s <= peak_index <= e]
    if not landing_at:
        return None
    landing_index = landing_at[0]

    row = {
        "Subject": subject.id, "Trial": trial.name,
        "Condition": trial.condition, "Side": side,
        "HasPushOff": landing_index > 0,
    }
    dt = float(np.median(np.diff(time)))
    phases = {"Landing": contacts[landing_index]}
    if landing_index > 0:
        phases["PushOff"] = contacts[landing_index - 1]

    for phase, (start, end) in phases.items():
        window = slice(start, end + 1)
        row[f"{phase}Duration_s"] = round((end - start + 1) * dt, 4)
        row[f"{phase}PeakGRF_N"] = round(float(np.nanmax(grf[window, 1])), 1)
        for label, moment, angle in (("Framework", framework_moment, framework_angle),
                                     ("Marker", marker_moment, marker_angle)):
            support = -np.sum(moment, axis=1)
            row.update(_phase_metrics(label, phase, moment, angle, support, time, window))
    return row


def main():
    subject = Subject(SUBJECT_ID, DEFAULT)
    activity = activity_for(ACTIVITY_NAME, DEFAULT)
    discovery = TrialDiscovery(DEFAULT)
    rows = []
    for condition in DEFAULT.conditions:
        for trial in discovery.trials(subject, activity, condition, for_analysis=True):
            try:
                row = _trial_row(subject, trial)
            except Exception as error:  # noqa: BLE001
                print(f"  {trial.name}: FAILED ({type(error).__name__}: {error})")
                continue
            if row is None:
                print(f"  {trial.name}: skipped (missing inputs or no landing contact)")
                continue
            rows.append(row)

    table = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "ab03_phase_kinetics.csv", index=False)
    print(f"\n{len(table)} trials -> {RESULTS / 'ab03_phase_kinetics.csv'}")
    print(f"trials with a captured push-off contact: {int(table.HasPushOff.sum())}/{len(table)}")
    pd.set_option("display.width", 240)
    show = ["Trial", "Side", "PushOffDuration_s", "LandingDuration_s",
            "Framework_PushOff_KneeConcWork_J", "Framework_Landing_KneeEccWork_J",
            "Framework_PushOff_HipConcWork_J", "Framework_PushOff_SupportPeak_Nm",
            "Framework_Landing_SupportPeak_Nm"]
    print(table[[c for c in show if c in table.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
