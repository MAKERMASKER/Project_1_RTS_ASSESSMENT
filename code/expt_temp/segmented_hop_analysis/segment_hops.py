"""segment_hops.py -- per-CYCLE segmentation and GRF analysis for lateral hop
and skater hop.

Why this exists
---------------
LH and SKH are multi-cycle activities, but production framework processing
analyses only ONE canonical event per trial (LateralHop's own docstring: "Only
this first single-support landing is analyzed"). Measured: LH/SKH framework
windows are 0.8-1.3 s with 1-2 contact bouts, versus a hopping bout that
contains many. So SLJ's one-row-per-trial model cannot express these
activities, and per-cycle segmentation is impossible from the existing
framework output.

This module widens the window to the whole bout and segments it into cycles.

**It does NOT touch production sync.** It calls `activity.synchronize()` to
obtain the validated shift (which for LH is the user's own manual per-trial
click from `lh_manual_sync`), then rebuilds the window as the full
insole<->OpenCap overlap instead of the single-event clip. Same anchor, wider
window.

Cycle definitions (as specified)
--------------------------------
- **SKH**: left -> right -> left is ONE complete cycle: push-off from left,
  landing on right, push-off from right, landing on left. Consecutive cycles
  share their middle landing. Each cycle contributes TWO stance phases (one
  per side), so a cycle yields an L row and an R row -- that is what makes an
  L-vs-R LSI possible for a bilateral activity.
- **LH**: one cycle is push-off -> land on the SAME leg, i.e. contact[k]
  through contact[k+1]. A cycle is ACCEPTED only if the distance between the
  two successive foot landings exceeds MIN_ML_LANDING_SEPARATION_M in the
  MEDIOLATERAL direction, measured from OpenCap marker data.

  ML direction is taken as the foot's own anatomical ML axis
  (`InsoleGeometry.frames()`'s middle column, averaged over the two landings),
  not a global axis -- global X/Z are not comparable between the marker lab
  and OpenCap (CLAUDE.md), whereas the foot's anatomical frame is built fresh
  from that trial's own markers and is frame-invariant.

Marker alignment
----------------
SLJ aligns marker to framework by matching the single GRF peak. Over a
multi-cycle bout that is fragile (many similar peaks), so this uses
cross-correlation of the whole vertical-force trace. The peak-based offset is
ALSO reported: the disagreement between the two is a direct, honest
sync-quality diagnostic, which is the thing to look at if segmentation looks
wrong.

Axis convention: marker Fx/Fz are reprojected into OpenCap's frame before any
comparison (`opencap_X = +marker_Z`, `opencap_Z = -marker_X`), exactly as
`outcomes.py` does. Fy needs no correction.

Scope of this pass: GRF (Fx/Fy/Fz) only. Joint moments and COP need the
inverse-dynamics stage over the widened window, which is deliberately deferred
until the segmentation itself is confirmed to be correct.

Run from code/:
    ..\\opensimPy\\Scripts\\python.exe expt_temp/segmented_hop_analysis/segment_hops.py
    ..\\opensimPy\\Scripts\\python.exe expt_temp/segmented_hop_analysis/segment_hops.py --subject AB08
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import signal  # noqa: E402

CODE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(CODE_ROOT))

import io_tables  # noqa: E402
from activities import activity_for  # noqa: E402
from calibration import InsoleCalibration  # noqa: E402
from config import DEFAULT  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from framework_processor import FrameworkProcessor  # noqa: E402
from insole_geometry import InsoleGeometry  # noqa: E402
from io_tables import interpolate_table, read_insole, read_storage, read_trc  # noqa: E402
from opensim_env import center_of_mass  # noqa: E402
from subject import Subject  # noqa: E402

SUBJECT_IDS = ("AB03", "AB05", "AB07", "AB08", "AB10", "NTS01")
ACTIVITIES = ("lateral_hop", "skater_hop")
RESULTS = Path(__file__).resolve().parent / "results"

MIN_ML_LANDING_SEPARATION_M = 0.40
# A genuine hop stance, reusing LateralHop's own validated duration signature
# (real hop contacts ~0.25-1.3 s; long double-support holds are >= 2.26 s).
MIN_STANCE_S = 0.15
MAX_STANCE_S = 1.8


def _full_bout(activity, trial, subject, trc, ik_all, insole_all):
    """Whole-bout window built on production's own validated sync shift."""
    _t, _i, _k, sync = activity.synchronize(trial, trc, ik_all, insole_all, subject.rules)
    shift = float(sync["shift_s"])
    shifted = insole_all.time.to_numpy(float) + shift
    low = max(float(trc["time"].min()), float(ik_all.time.min()))
    high = min(float(trc["time"].max()), float(ik_all.time.max()))
    indices = np.flatnonzero((shifted >= low) & (shifted <= high))
    if len(indices) < 50:
        raise ValueError(f"No usable whole-bout overlap for {trial.name}")
    use = slice(indices[0], indices[-1] + 1)
    time = shifted[use]
    return time, insole_all.iloc[use].reset_index(drop=True), sync


def _framework_forces(fp, subject, activity, trial, trc, ik_all, time, insole, calibration):
    padded_time, start, stop = fp._padded_time(time, ik_all)
    padded_ik = interpolate_table(ik_all, padded_time)
    com_padded, mass = center_of_mass(trial.opencap_model, padded_ik)
    com_padded = fp._smooth_com(com_padded, padded_time, DEFAULT.com_smoothing_alpha)
    pad_dt = float(np.median(np.diff(padded_time)))
    acceleration = np.column_stack([
        np.gradient(np.gradient(com_padded[:, axis], pad_dt), pad_dt) for axis in range(3)
    ])
    total = mass * (acceleration - np.array([0, -DEFAULT.gravity_mps2, 0]))
    geometry = InsoleGeometry.for_subject(subject)
    return fp._bilateral_forces(
        activity, trial, trc, geometry, time, insole, total[start:stop],
        calibration, subject.rules,
    )


def _marker_offset(time, framework_vertical, plate_time, plate_vertical):
    """(cross_correlation_offset, peak_offset) in seconds.

    `plate` sampled at `time + offset` lines up with the framework trace.
    Both are returned so their disagreement can be reported as a sync check.
    """
    dt = float(np.median(np.diff(time)))
    grid = np.arange(float(plate_time.min()), float(plate_time.max()), dt)
    resampled = np.interp(grid, plate_time, plate_vertical)
    a = resampled - resampled.mean()
    b = np.nan_to_num(framework_vertical - np.nanmean(framework_vertical))
    correlation = signal.correlate(a, b, mode="full")
    lags = signal.correlation_lags(len(a), len(b), mode="full")
    best = int(lags[int(np.argmax(correlation))])
    cross = float(grid[0] + best * dt - time[0])
    peak = float(plate_time.iloc[int(np.argmax(plate_vertical))] - time[int(np.nanargmax(framework_vertical))])
    return cross, peak


def _stances(mask, dt):
    """[start, end] index pairs of genuine hop stances."""
    out = []
    for start, end in io_tables.segments(np.asarray(mask, bool)):
        if MIN_STANCE_S <= (end - start + 1) * dt <= MAX_STANCE_S:
            out.append((int(start), int(end)))
    return out


def _skh_cycles(right_contact, left_contact, dt):
    """left -> right -> left; consecutive cycles share their middle landing."""
    events = ([("L", s, e) for s, e in _stances(left_contact, dt)]
              + [("R", s, e) for s, e in _stances(right_contact, dt)])
    events.sort(key=lambda item: item[1])
    cycles = []
    for index in range(len(events) - 2):
        first, middle, last = events[index], events[index + 1], events[index + 2]
        if first[0] == "L" and middle[0] == "R" and last[0] == "L":
            cycles.append({"start": first[1], "end": last[2],
                           "stances": [("L", first[1], first[2]), ("R", middle[1], middle[2])],
                           "accepted": True, "reject_reason": ""})
    return cycles


def _lh_cycles(own_contact, trc, time, side, dt):
    """push-off -> land on the same leg, gated on ML landing separation."""
    stances = _stances(own_contact, dt)
    centre = np.mean([InsoleGeometry.marker(trc, time, side, role)
                      for role in ("heel", "toe1", "toe5")], axis=0)
    _origin, rotation = InsoleGeometry.frames(trc, time, side)
    ml_axis = rotation[:, :, 1]

    cycles = []
    for index in range(len(stances) - 1):
        (start_a, end_a), (start_b, end_b) = stances[index], stances[index + 1]
        displacement = centre[start_b] - centre[start_a]
        axis = ml_axis[start_a] + ml_axis[start_b]
        norm = np.linalg.norm(axis)
        separation = float(np.abs(np.dot(displacement, axis / norm))) if norm > 0 else np.nan
        accepted = np.isfinite(separation) and separation > MIN_ML_LANDING_SEPARATION_M
        cycles.append({
            "start": start_a, "end": end_b,
            "stances": [(side, start_b, end_b)],  # the LANDING of this cycle
            "ml_separation_m": separation, "accepted": bool(accepted),
            "reject_reason": "" if accepted else f"ML separation {separation:.3f} m <= {MIN_ML_LANDING_SEPARATION_M}",
        })
    return cycles


def _analyse_trial(trial, subject, activity, fp, calibration):
    trc = read_trc(trial.effective_opencap_trc())
    ik_all = read_storage(trial.effective_opencap_ik())
    insole_all = read_insole(trial.insole)
    time, insole, sync = _full_bout(activity, trial, subject, trc, ik_all, insole_all)
    dt = float(np.median(np.diff(time)))

    (right_force, right_cop, right_contact,
     left_force, left_cop, left_contact) = _framework_forces(
        fp, subject, activity, trial, trc, ik_all, time, insole, calibration
    )

    force_plate = read_storage(trial.force_mot)
    anchor_side = trial.side if trial.side else "R"
    anchor_framework = (right_force if anchor_side == "R" else left_force)[:, 1]
    anchor_plate = np.abs(force_plate[f"ground_force_{1 if anchor_side == 'R' else 2}_vy"].to_numpy(float))
    cross_offset, peak_offset = _marker_offset(time, anchor_framework, force_plate.time, anchor_plate)

    marker = {}
    for side in ("R", "L"):
        number = 1 if side == "R" else 2
        native_x = np.interp(time + cross_offset, force_plate.time + 0,
                             force_plate[f"ground_force_{number}_vx"].to_numpy(float),
                             left=np.nan, right=np.nan)
        native_z = np.interp(time + cross_offset, force_plate.time + 0,
                             force_plate[f"ground_force_{number}_vz"].to_numpy(float),
                             left=np.nan, right=np.nan)
        marker[side] = np.column_stack([
            native_z,                       # opencap_X = +marker_Z
            np.interp(time + cross_offset, force_plate.time + 0,
                      np.abs(force_plate[f"ground_force_{number}_vy"].to_numpy(float)),
                      left=np.nan, right=np.nan),
            -native_x,                      # opencap_Z = -marker_X
        ])

    if activity.name == "skater_hop":
        cycles = _skh_cycles(right_contact, left_contact, dt)
    else:
        cycles = _lh_cycles(
            right_contact if trial.side == "R" else left_contact, trc, time, trial.side, dt
        )

    condition = "EXO" if "_EXO_" in trial.name else "NOEXO"
    rows = []
    for number, cycle in enumerate(cycles, start=1):
        for side, start, end in cycle["stances"]:
            framework = right_force if side == "R" else left_force
            window = slice(start, end + 1)
            truth = marker[side][window]
            estimate = framework[window]
            valid = np.isfinite(truth).all(axis=1) & np.isfinite(estimate).all(axis=1)
            if valid.sum() < 5:
                continue
            rows.append({
                "Subject": subject.id, "Activity": activity.name, "Trial": trial.name,
                "Condition": condition, "Side": side, "Cycle": number,
                "Accepted": cycle["accepted"], "RejectReason": cycle["reject_reason"],
                "MlSeparation_m": round(cycle.get("ml_separation_m", np.nan), 4),
                "StanceDuration_s": round((end - start + 1) * dt, 3),
                "PeakFrameworkGRF_N": round(float(np.nanmax(estimate[valid][:, 1])), 1),
                "PeakMarkerGRF_N": round(float(np.nanmax(truth[valid][:, 1])), 1),
                "GrfRmse_N": round(float(np.sqrt(np.mean((estimate[valid][:, 1] - truth[valid][:, 1]) ** 2))), 1),
                "GrfXRmse_N": round(float(np.sqrt(np.mean((estimate[valid][:, 0] - truth[valid][:, 0]) ** 2))), 1),
                "GrfZRmse_N": round(float(np.sqrt(np.mean((estimate[valid][:, 2] - truth[valid][:, 2]) ** 2))), 1),
                "SyncCrossCorr_s": round(cross_offset, 3),
                "SyncPeak_s": round(peak_offset, 3),
                "SyncDisagreement_s": round(abs(cross_offset - peak_offset), 3),
                "QualityOmit": bool(sync.get("quality_omit", False)),
            })

    _plot_bout(trial, activity, time, right_force, left_force, marker, cycles,
               right_contact, left_contact, cross_offset, peak_offset)
    return rows, cycles


def _plot_bout(trial, activity, time, right_force, left_force, marker, cycles,
               right_contact, left_contact, cross_offset, peak_offset):
    """Whole-bout framework-vs-marker vertical GRF with cycle segmentation
    drawn on -- the plot for judging whether segmentation and sync hold up."""
    out = RESULTS / "plots" / activity.name / trial.name.split("_")[0]
    out.mkdir(parents=True, exist_ok=True)
    relative = time - time[0]
    sides = ("R", "L") if activity.name == "skater_hop" else (trial.side,)

    figure, axes = plt.subplots(len(sides), 1, figsize=(15, 4.2 * len(sides)), squeeze=False)
    for row, side in enumerate(sides):
        axis = axes[row][0]
        framework = right_force if side == "R" else left_force
        contact = right_contact if side == "R" else left_contact
        axis.plot(relative, marker[side][:, 1], color="black", lw=1.3, label=f"Marker plate {side}")
        axis.plot(relative, framework[:, 1], color="red", lw=1.1, label=f"Framework {side}")
        axis.fill_between(relative, 0, axis.get_ylim()[1], where=contact,
                          color="grey", alpha=0.12, label="insole contact")
        for number, cycle in enumerate(cycles, start=1):
            colour = "green" if cycle["accepted"] else "red"
            axis.axvspan(relative[cycle["start"]], relative[cycle["end"]],
                         color=colour, alpha=0.10)
            axis.axvline(relative[cycle["start"]], color=colour, ls="--", lw=0.9)
            axis.text(relative[cycle["start"]], axis.get_ylim()[1] * 0.92, str(number),
                      fontsize=7, color=colour)
        axis.set_ylabel(f"{side} vertical GRF (N)")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7, loc="upper right")

    accepted = sum(1 for c in cycles if c["accepted"])
    axes[-1][0].set_xlabel("Time (s, from bout start)")
    figure.suptitle(
        f"{trial.name} -- {len(cycles)} cycles, {accepted} accepted "
        f"(green=accepted, red=rejected) | sync xcorr={cross_offset:.3f}s "
        f"peak={peak_offset:.3f}s disagree={abs(cross_offset - peak_offset):.3f}s"
    )
    figure.tight_layout()
    figure.savefig(out / f"{trial.name}_cycles.png", dpi=130)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject")
    args = parser.parse_args()
    subject_ids = (args.subject,) if args.subject else SUBJECT_IDS

    discovery = TrialDiscovery(DEFAULT)
    fp = FrameworkProcessor(DEFAULT)
    calibration_obj = InsoleCalibration(DEFAULT)
    all_rows = []

    for subject_id in subject_ids:
        subject = Subject(subject_id, DEFAULT)
        try:
            calibration = calibration_obj.load(subject)
        except FileNotFoundError:
            print(f"{subject_id}: no calibration -- skipping")
            continue
        for activity_name in ACTIVITIES:
            activity = activity_for(activity_name, DEFAULT)
            trials = [
                t for t in discovery.trials(subject, activity, "all", for_analysis=True)
                if t.insole.is_file() and t.force_mot.is_file()
                and t.effective_opencap_trc() and Path(t.effective_opencap_trc()).is_file()
                and t.effective_opencap_ik() and Path(t.effective_opencap_ik()).is_file()
            ]
            cycles_total = accepted_total = 0
            for trial in sorted(trials, key=lambda t: t.name):
                try:
                    rows, cycles = _analyse_trial(trial, subject, activity, fp, calibration)
                    all_rows.extend(rows)
                    cycles_total += len(cycles)
                    accepted_total += sum(1 for c in cycles if c["accepted"])
                except Exception as error:  # noqa: BLE001
                    print(f"  {trial.name}: FAILED ({type(error).__name__}: {error})")
            print(f"{subject_id}/{activity_name}: {len(trials)} trials, "
                  f"{cycles_total} cycles, {accepted_total} accepted")

    if not all_rows:
        print("No cycles produced.")
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(all_rows)
    table.to_csv(RESULTS / "cycle_metrics.csv", index=False)
    print(f"\n{len(table)} stance rows -> {RESULTS / 'cycle_metrics.csv'}")


if __name__ == "__main__":
    main()
