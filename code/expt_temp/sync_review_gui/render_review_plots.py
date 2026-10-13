"""render_review_plots.py -- pre-render one sync-review PNG per trial,
headlessly.

Why pre-render
--------------
The interactive version of this review (`review_sync.py`) froze repeatedly.
Measuring ruled out an artist leak, slow CSV writes and the prefetch, and
fixing a double-render brought each advance down to ~0.67 s -- and it still
froze. The remaining differences from the GUI that DOES work
(`state_machine_sync/select_sync_zoom.py`) were all things happening inside
the Tk event loop: OpenSim file reads, `activity.synchronize()`, a force-plate
cross-correlation, and finally a background thread calling OpenSim readers --
and OpenSim's C++ layer is not thread-safe, so that last one can deadlock the
process outright with no Python traceback.

So this does all of that work offline, with Agg, where a stall costs nothing
and is fully debuggable. `browse_review.py` then only loads PNGs, which cannot
block on OpenSim, threads or synchronisation.

Resumable: a trial whose PNG already exists is skipped, so repeated runs make
progress regardless of any time limit. Pass --overwrite to re-render.

Run from code/:
    ..\\opensimPy\\Scripts\\python.exe expt_temp/sync_review_gui/render_review_plots.py
    ..\\opensimPy\\Scripts\\python.exe expt_temp/sync_review_gui/render_review_plots.py --subject AB08
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import signal as scipy_signal  # noqa: E402

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent.parent
sys.path.insert(0, str(CODE_ROOT))

from activities import activity_for  # noqa: E402
from config import DEFAULT  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from insole_geometry import InsoleGeometry  # noqa: E402
from io_tables import read_insole, read_storage, read_trc  # noqa: E402
from subject import Subject  # noqa: E402

SUBJECT_IDS = ("AB03", "AB05", "AB07", "AB08", "AB10", "NTS01")
ACTIVITIES = ("single_leg_jump", "lateral_hop", "skater_hop", "triple_hop", "walk")
PLOTS = HERE / "plots"
INDEX = HERE / "results" / "render_index.csv"


def _thin(x, y, limit=4000):
    n = len(x)
    if n <= limit:
        return x, y
    step = int(np.ceil(n / limit))
    return x[::step], y[::step]


def _plate_offset(time, reference, plate_time, plate_values):
    dt = float(np.median(np.diff(time)))
    grid = np.arange(float(plate_time.min()), float(plate_time.max()), dt)
    if len(grid) < 4:
        return np.nan
    resampled = np.interp(grid, plate_time, plate_values)
    a = resampled - resampled.mean()
    b = np.nan_to_num(reference - np.nanmean(reference))
    if len(b) < 4:
        return np.nan
    correlation = scipy_signal.correlate(a, b, mode="full")
    lags = scipy_signal.correlation_lags(len(a), len(b), mode="full")
    return float(grid[0] + int(lags[int(np.argmax(correlation))]) * dt - time[0])


def _render(subject, activity, trial, path):
    """One review figure. Returns a dict of what the reviewer needs to know."""
    info = {"Subject": subject.id, "Activity": activity.name, "Trial": trial.name,
            "Shift_s": np.nan, "ShiftSource": "", "QualityOmit": False, "Error": ""}
    trc = read_trc(trial.effective_opencap_trc())
    insole = read_insole(trial.insole)
    side = trial.side if trial.side else "R"

    opencap_time = np.asarray(trc["time"], float)
    heel = InsoleGeometry.marker(trc, opencap_time, side, "heel")
    toe1 = InsoleGeometry.marker(trc, opencap_time, side, "toe1")
    toe5 = InsoleGeometry.marker(trc, opencap_time, side, "toe5")
    _origin, rotation = InsoleGeometry.frames(trc, opencap_time, side)
    vertical = rotation[:, :, 2]
    lowest = np.minimum((heel - 0.03 * vertical)[:, 1],
                        (0.5 * (toe1 + toe5) - 0.03 * vertical)[:, 1])
    finite = np.isfinite(lowest)
    sole = lowest - (np.nanpercentile(lowest[finite], 5) if finite.any() else 0.0)
    pitch = np.degrees(np.arccos(np.clip(vertical[:, 1], -1, 1)))

    insole_time = insole.time.to_numpy(float)
    force = insole["right_force" if side == "R" else "left_force"].to_numpy(float)

    try:
        _t, _i, _k, sync = activity.synchronize(
            trial, trc, read_storage(trial.effective_opencap_ik()), insole, subject.rules)
        info["Shift_s"] = float(sync["shift_s"])
        info["ShiftSource"] = str(sync.get("method", "activity.synchronize"))[:40]
        info["QualityOmit"] = bool(sync.get("quality_omit", False))
    except Exception as error:  # noqa: BLE001 -- a trial without a sync still renders
        info["ShiftSource"] = f"UNAVAILABLE: {type(error).__name__}"
        info["Error"] = str(error)[:120]

    figure, axes = plt.subplots(2, 1, figsize=(16, 8))
    pitch_axis = axes[0].twinx()
    axes[0].plot(*_thin(opencap_time, 100 * sole), color="darkorange", lw=1.0,
                 label="sole height (cm)")
    axes[0].set_ylim(-2, 45)
    axes[0].set_ylabel("sole height (cm)")
    pitch_axis.plot(*_thin(opencap_time, pitch), color="purple", lw=0.9, alpha=0.8,
                    label="foot pitch (deg)")
    pitch_axis.set_ylabel("pitch (deg)", color="purple")
    axes[0].set_title(f"OpenCap kinematics (side {side})")

    shift = info["Shift_s"]
    if np.isfinite(shift):
        shifted = insole_time + shift
        axes[1].plot(*_thin(shifted, force), color="deepskyblue", lw=1.0,
                     label="insole (shifted by current sync)")
        if trial.force_mot and Path(trial.force_mot).is_file():
            force_plate = read_storage(trial.force_mot)
            column = f"ground_force_{1 if side == 'R' else 2}_vy"
            if column in force_plate:
                plate_time = force_plate.time.to_numpy(float)
                plate = np.abs(force_plate[column].to_numpy(float))
                aligned = np.interp(opencap_time, shifted, force, left=np.nan, right=np.nan)
                offset = _plate_offset(opencap_time, aligned, plate_time, plate)
                if np.isfinite(offset):
                    axes[1].plot(*_thin(opencap_time,
                                        np.interp(opencap_time + offset, plate_time, plate,
                                                  left=np.nan, right=np.nan)),
                                 color="black", lw=1.0, alpha=0.75,
                                 label="force plate (independent check)")
        low = max(float(opencap_time.min()), float(shifted.min()))
        high = min(float(opencap_time.max()), float(shifted.max()))
        for axis in axes:
            axis.set_xlim(low, high)
        axes[1].set_xlabel("OpenCap clock (s)  --  contacts should line up with the panel above")
    else:
        axes[1].plot(*_thin(insole_time, force), color="deepskyblue", lw=1.0, label="insole")
        axes[1].set_xlabel("insole clock (s)  --  NO SYNC AVAILABLE, panels are NOT aligned")
    axes[1].set_ylabel("force (N)")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8, loc="upper right")

    flags = []
    if info["QualityOmit"]:
        flags.append("quality_omit")
    if info["Error"]:
        flags.append(info["ShiftSource"])
    title = (f"{trial.name}   ({activity.name}, side {side})   "
             + (f"shift={shift:+.3f}s [{info['ShiftSource']}]" if np.isfinite(shift)
                else "NO SYNC"))
    figure.suptitle(title + ("   !! " + ", ".join(flags) if flags else ""),
                    color="red" if flags else "black", fontsize=11)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=100)
    plt.close(figure)
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject")
    parser.add_argument("--activity")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    subject_ids = (args.subject,) if args.subject else SUBJECT_IDS
    activities = (args.activity,) if args.activity else ACTIVITIES

    discovery = TrialDiscovery(DEFAULT)
    existing = {}
    if INDEX.is_file():
        existing = {r["Trial"]: r for r in pd.read_csv(INDEX).to_dict("records")}

    rows, made, skipped, failed = [], 0, 0, 0
    for subject_id in subject_ids:
        subject = Subject(subject_id, DEFAULT)
        for activity_name in activities:
            activity = activity_for(activity_name, DEFAULT)
            for trial in sorted(discovery.trials(subject, activity, "all", for_analysis=True),
                                key=lambda t: t.name):
                path = PLOTS / activity_name / subject_id / f"{trial.name}.png"
                if path.is_file() and not args.overwrite and trial.name in existing:
                    rows.append(existing[trial.name])
                    skipped += 1
                    continue
                try:
                    info = _render(subject, activity, trial, path)
                    info["Plot"] = str(path.relative_to(HERE))
                    rows.append(info)
                    made += 1
                except Exception as error:  # noqa: BLE001
                    print(f"  {trial.name}: FAILED ({type(error).__name__}: {error})")
                    rows.append({"Subject": subject_id, "Activity": activity_name,
                                 "Trial": trial.name, "Shift_s": np.nan,
                                 "ShiftSource": "RENDER_FAILED", "QualityOmit": True,
                                 "Error": f"{type(error).__name__}: {error}"[:120],
                                 "Plot": ""})
                    failed += 1
            print(f"{subject_id}/{activity_name}: rendered={made} skipped={skipped} failed={failed}",
                  flush=True)

    INDEX.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(INDEX, index=False)
    print(f"\nRENDER_DONE  rendered={made} skipped={skipped} failed={failed}")
    print(f"-> {PLOTS}\n-> {INDEX}")


if __name__ == "__main__":
    main()
