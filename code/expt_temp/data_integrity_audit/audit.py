"""audit.py -- data-integrity sweep over every subject, activity and trial.

Looks for trials that are obviously wrong rather than merely inaccurate:
implausible forces, extreme framework-vs-marker disagreement, bad sync, NaN or
constant signals, non-physical loading rates, out-of-range kinematics, COP
teleports, and model-mass mismatches.

Trials are enumerated through `TrialDiscovery`, NOT by globbing the results
tree: `results/<S>/framework/<activity>/` also contains stale `local_original`
and `local_repaired` directories left by trc_repair, so a filesystem count
overstates the trial set (AB03 lateral_hop has 21 npz files for 7 real trials).

Every threshold below is deliberately GENEROUS -- the goal is to surface
trials that are clearly broken, not to rank trials by quality. A flag means
"look at this", not "this is wrong".

Axis note: framework-vs-marker vertical force (Fy) needs no frame correction.
Fx/Fz do -- the marker lab and OpenCap global frames are ~90 deg apart
(CLAUDE.md), so marker Fx/Fz are reprojected (`opencap_X = +marker_Z`,
`opencap_Z = -marker_X`) exactly as outcomes.py does before any comparison.

Run from code/:
    ..\\opensimPy\\Scripts\\python.exe expt_temp/data_integrity_audit/audit.py
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

SUBJECT_IDS = ("AB03", "AB05", "AB07", "AB08", "AB10", "NTS01")
ACTIVITIES = ("single_leg_jump", "lateral_hop", "skater_hop", "triple_hop")
RESULTS = Path(__file__).resolve().parent / "results"

# --- thresholds, all deliberately generous -------------------------------
MIN_PEAK_BW = 0.8          # a jump landing under 0.8 body weights is suspect
MAX_PEAK_BW = 6.0          # above ~6 BW is beyond reported single-leg landings
PEAK_RATIO_LOW = 0.60      # framework/marker peak disagreement bounds
PEAK_RATIO_HIGH = 1.70
MAX_GRF_RMSE_PCT = 35.0    # vs marker peak
MAX_NAN_FRACTION = 0.02
MAX_LOADING_RATE_BW_S = 800.0
MAX_KNEE_MOMENT_NM = 400.0
MAX_HIP_MOMENT_NM = 650.0
KNEE_ANGLE_RANGE = (-15.0, 170.0)
MAX_COP_JUMP_M = 0.30
MASS_TOLERANCE_PCT = 2.0
MIN_CONTACT_S = 0.08


def _issue(issues, trial, subject, activity, severity, kind, detail):
    issues.append({"Subject": subject.id, "Activity": activity.name, "Trial": trial.name,
                   "Severity": severity, "Issue": kind, "Detail": detail})


def _audit_trial(subject, activity, trial, issues):
    row = {"Subject": subject.id, "Activity": activity.name, "Trial": trial.name,
           "Condition": trial.condition, "Side": trial.side}
    package = trial.output / "trial_data.npz"
    if not package.is_file():
        _issue(issues, trial, subject, activity, "HIGH", "NO_FRAMEWORK_OUTPUT",
               "trial_data.npz missing although discovery accepts the trial")
        row["Status"] = "no_framework_output"
        return row

    data = np.load(package, allow_pickle=True)
    time = data["time"]
    grf = data["grf_N"]
    cop = data["cop_m"]
    contact = data["contact"].astype(bool)
    mass = float(data["mass_kg"])
    weight = mass * DEFAULT.gravity_mps2
    row["Status"] = "ok"
    row["Mass_kg"] = round(mass, 2)
    row["Frames"] = len(time)
    row["ContactFrames"] = int(contact.sum())
    row["QualityOmit"] = bool(data.get("quality_omit", False))

    if row["QualityOmit"]:
        _issue(issues, trial, subject, activity, "INFO", "QUALITY_OMIT",
               "already flagged quality_omit by the activity's own sync")

    # model mass vs measured mass
    if subject.mass_kg:
        drift = 100 * abs(mass - subject.mass_kg) / subject.mass_kg
        row["MassDrift_pct"] = round(drift, 2)
        if drift > MASS_TOLERANCE_PCT:
            _issue(issues, trial, subject, activity, "MEDIUM", "MODEL_MASS_MISMATCH",
                   f"model {mass:.1f} kg vs metadata {subject.mass_kg:.1f} kg ({drift:.1f}%)")

    # NaN / constant signals
    nan_fraction = float(np.mean(~np.isfinite(grf).all(axis=1)))
    row["NaNFraction"] = round(nan_fraction, 4)
    if nan_fraction > MAX_NAN_FRACTION:
        _issue(issues, trial, subject, activity, "HIGH", "NAN_IN_GRF",
               f"{100 * nan_fraction:.1f}% of GRF frames non-finite")
    if contact.sum() > 5:
        vertical_contact = grf[contact, 1]
        if np.nanstd(vertical_contact) < 1.0:
            _issue(issues, trial, subject, activity, "HIGH", "CONSTANT_GRF",
                   "vertical GRF has ~zero variance during contact")

    if contact.sum() < 5:
        _issue(issues, trial, subject, activity, "HIGH", "NO_CONTACT",
               f"only {int(contact.sum())} contact frames detected")
        return row

    # peak force in body weights
    peak = float(np.nanmax(grf[:, 1]))
    row["PeakFrameworkGRF_N"] = round(peak, 1)
    row["PeakFramework_BW"] = round(peak / weight, 2)
    if not (MIN_PEAK_BW <= peak / weight <= MAX_PEAK_BW):
        _issue(issues, trial, subject, activity, "HIGH", "IMPLAUSIBLE_PEAK_FORCE",
               f"framework peak {peak:.0f} N = {peak / weight:.2f} BW "
               f"(expected {MIN_PEAK_BW}-{MAX_PEAK_BW})")

    # loading rate
    dt = float(np.median(np.diff(time)))
    rate = np.abs(np.diff(grf[:, 1])) / dt / weight
    row["MaxLoadingRate_BWs"] = round(float(np.nanmax(rate)), 1)
    if np.nanmax(rate) > MAX_LOADING_RATE_BW_S:
        _issue(issues, trial, subject, activity, "MEDIUM", "EXTREME_LOADING_RATE",
               f"max |dF/dt| = {np.nanmax(rate):.0f} BW/s (threshold {MAX_LOADING_RATE_BW_S})")

    # contact bout durations
    bouts = segments(contact)
    durations = (bouts[:, 1] - bouts[:, 0] + 1) * dt
    row["ContactBouts"] = len(bouts)
    if len(durations) and durations.max() < MIN_CONTACT_S:
        _issue(issues, trial, subject, activity, "MEDIUM", "CONTACT_TOO_SHORT",
               f"longest contact only {durations.max():.3f} s")

    # COP teleports during contact. `framework_processor` ZEROES the COP
    # outside contact, so a step measured across a contact transition is a
    # jump from (0,0,0) to a real lab position -- ~1 m, and an artefact of the
    # zeroing, not a real discontinuity. (Measured: median max-step was 1.02 m
    # across all trials before this guard, flagging 131 trials.) Require both
    # frames of the step to be in contact, and erode one frame either side of
    # every transition.
    interior = contact.copy()
    transitions = np.flatnonzero(np.diff(contact.astype(int)) != 0)
    for index in transitions:
        interior[max(0, index - 1):min(len(interior), index + 3)] = False
    cop_step = np.linalg.norm(np.diff(cop, axis=0), axis=1)
    both_in_contact = interior[1:] & interior[:-1]
    valid_step = cop_step[both_in_contact & np.isfinite(cop_step)]
    if len(valid_step):
        row["MaxCopJump_m"] = round(float(np.nanmax(valid_step)), 4)
        if np.nanmax(valid_step) > MAX_COP_JUMP_M:
            _issue(issues, trial, subject, activity, "MEDIUM", "COP_DISCONTINUITY",
                   f"COP moves {np.nanmax(valid_step):.2f} m in one frame during contact")

    # kinematics range
    ik_names = [str(n) for n in data["ik_names"]]
    ik = pd.DataFrame(data["ik_values"], columns=ik_names)
    side = trial.side if trial.side else "R"
    knee = f"knee_angle_{side.lower()}"
    if knee in ik:
        low, high = float(np.nanmin(ik[knee])), float(np.nanmax(ik[knee]))
        row["KneeAngleMin_deg"] = round(low, 1)
        row["KneeAngleMax_deg"] = round(high, 1)
        if low < KNEE_ANGLE_RANGE[0] or high > KNEE_ANGLE_RANGE[1]:
            _issue(issues, trial, subject, activity, "MEDIUM", "KNEE_ANGLE_OUT_OF_RANGE",
                   f"knee angle spans {low:.0f}..{high:.0f} deg")

    # --- marker comparison -------------------------------------------
    marker_root = DEFAULT.results_root / subject.id / "marker_based" / trial.condition
    marker_id_path = marker_root / "id" / f"{trial.name}_inverse_dynamics.sto"
    framework_id = trial.output / f"{trial.name}_opencap_insole_id.sto"
    if not trial.force_mot or not Path(trial.force_mot).is_file():
        _issue(issues, trial, subject, activity, "MEDIUM", "NO_FORCE_PLATE",
               "ground-truth force MOT missing")
        return row

    force_plate = read_storage(trial.force_mot)
    plate = 1 if side == "R" else 2
    marker_vertical = np.abs(force_plate[f"ground_force_{plate}_vy"].to_numpy(float))
    shift = float(time[np.argmax(data["force"])] - force_plate.time.iloc[np.argmax(marker_vertical)])
    aligned = np.interp(time, force_plate.time + shift, marker_vertical, left=np.nan, right=np.nan)
    shared = contact & np.isfinite(aligned) & (aligned >= DEFAULT.contact_force_n)
    if shared.sum() < 10:
        _issue(issues, trial, subject, activity, "MEDIUM", "NO_SHARED_CONTACT",
               f"only {int(shared.sum())} frames where both framework and plate are loaded")
        return row

    marker_peak = float(np.nanmax(aligned[shared]))
    row["PeakMarkerGRF_N"] = round(marker_peak, 1)
    row["PeakMarker_BW"] = round(marker_peak / weight, 2)
    ratio = peak / marker_peak if marker_peak else np.nan
    row["PeakRatio_FwOverMk"] = round(ratio, 3)
    if np.isfinite(ratio) and not (PEAK_RATIO_LOW <= ratio <= PEAK_RATIO_HIGH):
        _issue(issues, trial, subject, activity, "HIGH", "EXTREME_PEAK_DISAGREEMENT",
               f"framework/marker peak ratio {ratio:.2f} "
               f"({peak:.0f} N vs {marker_peak:.0f} N)")
    if not (MIN_PEAK_BW <= marker_peak / weight <= MAX_PEAK_BW):
        _issue(issues, trial, subject, activity, "HIGH", "IMPLAUSIBLE_MARKER_PEAK",
               f"marker peak {marker_peak:.0f} N = {marker_peak / weight:.2f} BW")

    rmse = float(np.sqrt(np.nanmean((grf[shared, 1] - aligned[shared]) ** 2)))
    row["GrfRmse_N"] = round(rmse, 1)
    row["GrfRmse_pct"] = round(100 * rmse / marker_peak, 1) if marker_peak else np.nan
    if marker_peak and 100 * rmse / marker_peak > MAX_GRF_RMSE_PCT:
        _issue(issues, trial, subject, activity, "HIGH", "EXTREME_GRF_DISAGREEMENT",
               f"vertical GRF RMSE {rmse:.0f} N = {100 * rmse / marker_peak:.0f}% of marker peak")

    # correlation as a sync proxy
    both = shared & np.isfinite(grf[:, 1])
    if both.sum() > 20:
        correlation = float(np.corrcoef(grf[both, 1], aligned[both])[0, 1])
        row["GrfCorrelation"] = round(correlation, 3)
        if correlation < 0.5:
            _issue(issues, trial, subject, activity, "HIGH", "POOR_SYNC_OR_SHAPE",
                   f"framework-vs-plate vertical GRF correlation only {correlation:.2f}")

    # joint moments
    for path, label in ((framework_id, "Framework"), (marker_id_path, "Marker")):
        if not Path(path).is_file():
            _issue(issues, trial, subject, activity, "MEDIUM", f"NO_{label.upper()}_ID",
                   f"{label} inverse-dynamics file missing")
            continue
        table = read_storage(path)
        for joint, limit in (("knee_angle", MAX_KNEE_MOMENT_NM), ("hip_flexion", MAX_HIP_MOMENT_NM)):
            column = f"{joint}_{side.lower()}_moment"
            if column not in table:
                continue
            value = float(np.nanmax(np.abs(table[column].to_numpy(float))))
            row[f"{label}Peak{joint.split('_')[0].capitalize()}Moment_Nm"] = round(value, 1)
            if value > limit:
                _issue(issues, trial, subject, activity, "HIGH", "IMPLAUSIBLE_JOINT_MOMENT",
                       f"{label} peak |{joint}| = {value:.0f} Nm (limit {limit:.0f})")
    return row


def main():
    discovery = TrialDiscovery(DEFAULT)
    issues, rows = [], []
    for subject_id in SUBJECT_IDS:
        subject = Subject(subject_id, DEFAULT)
        for activity_name in ACTIVITIES:
            activity = activity_for(activity_name, DEFAULT)
            trials = discovery.trials(subject, activity, "all", for_analysis=True)
            for trial in sorted(trials, key=lambda t: t.name):
                try:
                    rows.append(_audit_trial(subject, activity, trial, issues))
                except Exception as error:  # noqa: BLE001
                    issues.append({"Subject": subject_id, "Activity": activity_name,
                                   "Trial": trial.name, "Severity": "HIGH",
                                   "Issue": "AUDIT_CRASHED",
                                   "Detail": f"{type(error).__name__}: {error}"})
            print(f"{subject_id}/{activity_name}: {len(trials)} trials audited")

    RESULTS.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(rows)
    audit.to_csv(RESULTS / "trial_audit.csv", index=False)
    problems = pd.DataFrame(issues)
    if len(problems):
        order = {"HIGH": 0, "MEDIUM": 1, "INFO": 2}
        problems["rank"] = problems.Severity.map(order)
        problems = problems.sort_values(["rank", "Issue", "Subject", "Trial"]).drop(columns="rank")
    problems.to_csv(RESULTS / "issues.csv", index=False)

    print(f"\n{len(audit)} trials audited, {len(problems)} issues -> {RESULTS}")
    if not len(problems):
        print("No issues flagged.")
        return
    print("\n=== issue counts by type and severity ===")
    print(problems.groupby(["Severity", "Issue"]).size().to_string())
    print("\n=== issues per subject/activity ===")
    print(problems[problems.Severity != "INFO"]
          .groupby(["Subject", "Activity"]).size().to_string())


if __name__ == "__main__":
    main()
