"""extract_moments.py -- AB03 SLJ per-trial EXTENSOR and FLEXOR joint kinetics,
separated by sign, for both the framework and marker pipelines.

Why this exists
---------------
`outcomes.py` stores joint peaks as `np.nanmax(np.abs(moment))`, which throws
the sign away. The isokinetic reference now reports quadriceps (extension) and
hamstring (flexion) separately, plus H:Q ratios, so the movement side has to
distinguish them too. That requires going back to the signed inverse-dynamics
traces.

Sign convention: OpenSim reports generalized forces on the FLEXION coordinate
(`knee_angle`, `hip_flexion`), so a NEGATIVE moment is an EXTENSOR moment and a
positive one is a flexor moment. (Same convention `outcomes.py` relies on for
its support moment, where it negates the sum.) Verified on real AB03 data:
knee moment runs to -178 Nm during landing, which must be the extensor peak.

Alignment and windowing replicate `OutcomeExtractor.analyze_trial()` exactly --
the same peak-GRF shift, the same landing segment (the marker-confirmed contact
containing the peak) -- so these numbers sit on the same basis as the existing
trial_summary, rather than being a parallel definition that happens to differ.

Hamstring analogues, stated honestly: in a single-leg jump the knee moment is
extensor-dominated, so a knee FLEXOR moment is small and intermittent. The
hamstrings are biarticular, acting as hip EXTENSORS as well as knee flexors,
and in a jump that is their major role. Both are therefore reported --
knee-flexor for the direct analogue, hip-extensor for the functional one --
and neither is presented as the single correct mapping.

Run from code/:
    ..\\opensimPy\\Scripts\\python.exe expt_temp/AB03_FullData_Iso_association/extract_moments.py
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


def _interp(source_time, values, target_time):
    return np.interp(target_time, source_time, values, left=np.nan, right=np.nan)


def _split(moment):
    """(extensor, flexor) magnitudes, both non-negative.

    Negative generalized force on a flexion coordinate == extensor moment.
    """
    return np.maximum(-moment, 0.0), np.maximum(moment, 0.0)


def _trial_row(subject, trial):
    package = trial.output / "trial_data.npz"
    framework_id = trial.output / f"{trial.name}_opencap_insole_id.sto"
    marker_root = DEFAULT.results_root / subject.id / "marker_based" / trial.condition
    marker_ik_path = marker_root / "ik" / f"{trial.name}_ik.mot"
    marker_id_path = marker_root / "id" / f"{trial.name}_inverse_dynamics.sto"
    required = (package, framework_id, marker_ik_path, marker_id_path, trial.force_mot)
    if not all(Path(p).is_file() for p in required):
        return None

    data = np.load(package, allow_pickle=True)
    if bool(data.get("quality_omit", False)):
        return None
    time = data["time"]
    force = data["force"]
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
    shift = float(time[np.argmax(force)] - force_plate.time.iloc[np.argmax(marker_vertical)])
    marker_force = _interp(force_plate.time + shift, marker_vertical, time)

    suffix = side.lower()
    angles = [f"hip_flexion_{suffix}", f"knee_angle_{suffix}", f"ankle_angle_{suffix}"]
    moments = [f"{a}_moment" for a in angles]
    framework_angle = np.column_stack([framework_ik[a] for a in angles])
    marker_angle = np.column_stack([
        _interp(marker_ik_table.time + shift, marker_ik_table[a], time) for a in angles
    ])
    framework_moment = np.column_stack([
        np.interp(time, framework_id_table.time, framework_id_table[m]) for m in moments
    ])
    marker_moment = np.column_stack([
        _interp(marker_id_table.time + shift, marker_id_table[m], time) for m in moments
    ])

    # Landing window: the marker-confirmed contact segment containing the peak
    # vertical GRF -- identical to OutcomeExtractor.analyze_trial().
    peak_index = int(np.argmax(grf[:, 1]))
    candidates = segments(contact & (marker_force >= DEFAULT.contact_force_n))
    landing_segment = candidates[
        (candidates[:, 0] <= peak_index) & (candidates[:, 1] >= peak_index)
    ]
    if not len(landing_segment):
        return None
    landing = np.zeros(len(time), bool)
    landing[landing_segment[0, 0]:landing_segment[0, 1] + 1] = True

    row = {
        "Subject": subject.id, "Trial": trial.name,
        "Condition": trial.condition, "Side": side,
        "LandingFrames": int(landing.sum()),
        "LandingDuration_s": round(float(landing.sum()) * float(np.median(np.diff(time))), 4),
    }

    for label, moment_set, angle_set in (("Framework", framework_moment, framework_angle),
                                         ("Marker", marker_moment, marker_angle)):
        velocity = np.gradient(np.deg2rad(angle_set), time, axis=0)
        power = moment_set * velocity
        for index, joint in enumerate(("Hip", "Knee", "Ankle")):
            extensor, flexor = _split(moment_set[:, index])
            window_time = time[landing]
            row[f"{label}{joint}ExtPeak_Nm"] = round(float(np.nanmax(extensor[landing])), 2)
            row[f"{label}{joint}FlexPeak_Nm"] = round(float(np.nanmax(flexor[landing])), 2)
            row[f"{label}{joint}ExtImpulse_Nms"] = round(
                float(np.trapezoid(extensor[landing], window_time)), 3)
            row[f"{label}{joint}FlexImpulse_Nms"] = round(
                float(np.trapezoid(flexor[landing], window_time)), 3)
            # Work split by the sign of the MOMENT, so "extensor work" is work
            # done while an extensor moment acts (concentric or eccentric),
            # matching how the isokinetic test attributes work to a muscle group.
            joint_power = power[landing, index]
            extensor_mask = moment_set[landing, index] < 0
            row[f"{label}{joint}ExtWork_J"] = round(
                float(np.trapezoid(np.abs(np.where(extensor_mask, joint_power, 0.0)), window_time)), 2)
            row[f"{label}{joint}FlexWork_J"] = round(
                float(np.trapezoid(np.abs(np.where(~extensor_mask, joint_power, 0.0)), window_time)), 2)

        # H:Q analogues -- knee flexor over knee extensor (direct), and hip
        # extensor over knee extensor (functional hamstring role in a jump).
        knee_ext = row[f"{label}KneeExtPeak_Nm"]
        knee_flex = row[f"{label}KneeFlexPeak_Nm"]
        hip_ext = row[f"{label}HipExtPeak_Nm"]
        row[f"{label}HQ_PeakRatio_pct"] = round(100 * knee_flex / knee_ext, 2) if knee_ext else np.nan
        row[f"{label}HipExtOverKneeExt_pct"] = round(100 * hip_ext / knee_ext, 2) if knee_ext else np.nan
        knee_ext_w = row[f"{label}KneeExtWork_J"]
        knee_flex_w = row[f"{label}KneeFlexWork_J"]
        row[f"{label}HQ_WorkRatio_pct"] = round(100 * knee_flex_w / knee_ext_w, 2) if knee_ext_w else np.nan
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
                print(f"  {trial.name}: skipped (missing inputs or no landing window)")
                continue
            rows.append(row)
    if not rows:
        print("No trials extracted.")
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "ab03_trial_kinetics.csv", index=False)
    print(f"{len(table)} trials -> {RESULTS / 'ab03_trial_kinetics.csv'}")
    print(table.groupby(["Condition", "Side"]).size().to_string())
    print()
    show = ["Trial", "Side", "FrameworkKneeExtPeak_Nm", "MarkerKneeExtPeak_Nm",
            "FrameworkKneeFlexPeak_Nm", "MarkerKneeFlexPeak_Nm",
            "FrameworkHipExtPeak_Nm", "MarkerHipExtPeak_Nm", "FrameworkHQ_PeakRatio_pct"]
    pd.set_option("display.width", 220)
    print(table[show].to_string(index=False))


if __name__ == "__main__":
    main()
