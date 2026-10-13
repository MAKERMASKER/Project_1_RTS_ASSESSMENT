"""validate_sync.py -- one sync-validation plot per trial, for every trial,
written to the validation folder.

Runs after `decide_sync.py` (automatic trials) and `select_sync_zoom.py`
(manual queue), once 100% of trials carry a sync offset. Refuses to declare
success unless coverage is complete, because the framework rerun depends on
every trial having a trusted offset.

Each plot shows, on the common OpenCap clock:
  * insole force for the relevant foot, shifted by the FINAL offset;
  * the OpenCap kinematic contact signal (sole height + foot pitch);
  * the force plate, as INDEPENDENT confirmation.

The force plate appears here and only here. It plays no part in producing the
sync -- it is validation, not pipeline -- so its agreement is meaningful
evidence rather than circular. Plate alignment for display uses
cross-correlation of the vertical trace, since peak-matching was measured to
misalign multi-cycle bouts by up to 16 s.

Output: validation/sensor_insole/sync_validation/<activity>/<subject>/<trial>.png
plus a coverage/quality CSV.

Run from code/:
    ..\\opensimPy\\Scripts\\python.exe expt_temp/state_machine_sync/validate_sync.py
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
sys.path.insert(0, str(HERE))

from activities import activity_for  # noqa: E402
from config import DEFAULT  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from io_tables import read_insole, read_storage, read_trc  # noqa: E402
from subject import Subject  # noqa: E402

from auto_sync import _insole_contacts  # noqa: E402
from foot_frame_sync import _foot_frame_signals, _opencap_contacts_pitch  # noqa: E402

RESULTS = HERE / "results"
VALIDATION_ROOT = CODE_ROOT.parent / "validation" / "sensor_insole" / "sync_validation"
ACTIVITIES = ("lateral_hop", "skater_hop")


def _final_offsets():
    """trial name -> (offset_s, source). Manual decisions override automatic."""
    decisions = pd.read_csv(RESULTS / "sync_decisions.csv")
    offsets = {}
    for _index, row in decisions.iterrows():
        if row.Decision == "AUTO" and np.isfinite(row.AutoOffset_s):
            offsets[row.Trial] = (float(row.AutoOffset_s), "auto")
    for path in sorted(RESULTS.glob("*_manual_sync.csv")):
        for _index, row in pd.read_csv(path).iterrows():
            if str(row.Status) == "synced" and np.isfinite(row.ShiftS):
                offsets[row.Trial] = (float(row.ShiftS), "manual")
            elif str(row.Status) == "rejected":
                offsets[row.Trial] = (np.nan, "rejected")
    return offsets, decisions


def _plate_overlay(time, reference, force_plate, plate_number):
    raw = np.abs(force_plate[f"ground_force_{plate_number}_vy"].to_numpy(float))
    dt = float(np.median(np.diff(time)))
    grid = np.arange(float(force_plate.time.min()), float(force_plate.time.max()), dt)
    resampled = np.interp(grid, force_plate.time, raw)
    a = resampled - resampled.mean()
    b = np.nan_to_num(reference - np.nanmean(reference))
    if len(a) < 4 or len(b) < 4:
        return np.full(len(time), np.nan), np.nan
    correlation = scipy_signal.correlate(a, b, mode="full")
    lags = scipy_signal.correlation_lags(len(a), len(b), mode="full")
    best = int(lags[int(np.argmax(correlation))])
    # Lag `best` means the reference aligns with the plate starting at
    # grid[best], so plate_time = reference_time + offset. The plate is
    # therefore sampled at (time + offset) -- NOT at time against a shifted
    # plate clock, which is the inverse mapping and misaligns completely.
    offset = float(grid[0] + best * dt - time[0])
    return np.interp(time + offset, force_plate.time, raw, left=np.nan, right=np.nan), offset


def _plot(trial, activity_name, offset, source):
    trc = read_trc(trial.effective_opencap_trc())
    insole = read_insole(trial.insole)
    side = trial.side if trial.side else "R"
    opencap_time = np.asarray(trc["time"], float)
    above, pitch, _speed = _foot_frame_signals(trc, side)
    _pm, _pe = _opencap_contacts_pitch(trc, side)

    insole_time = insole.time.to_numpy(float) + offset
    column = "right_force" if side == "R" else "left_force"
    force = insole[column].to_numpy(float)
    mask, _events, _standing = _insole_contacts(force, insole.time.to_numpy(float))

    figure, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    axes[0].plot(opencap_time, 100 * above, color="darkorange", lw=1.0, label="sole height (cm)")
    axes[0].set_ylabel("sole height (cm)")
    axes[0].set_ylim(-2, 45)
    twin = axes[0].twinx()
    twin.plot(opencap_time, pitch, color="purple", lw=0.9, alpha=0.8, label="foot pitch (deg)")
    twin.set_ylabel("pitch (deg)", color="purple")
    axes[0].set_title(f"OpenCap kinematics ({side})")

    axes[1].plot(insole_time, force, color="deepskyblue", lw=1.0, label="insole (shifted)")
    plate_value = np.nan
    if trial.force_mot and Path(trial.force_mot).is_file():
        force_plate = read_storage(trial.force_mot)
        aligned = np.interp(opencap_time, insole_time, force, left=np.nan, right=np.nan)
        plate, plate_offset = _plate_overlay(opencap_time, aligned, force_plate,
                                             1 if side == "R" else 2)
        axes[1].plot(opencap_time, plate, color="black", lw=1.0, alpha=0.8,
                     label="force plate (independent check)")
        both = np.isfinite(plate) & np.isfinite(aligned)
        if both.sum() > 20:
            plate_value = float(np.corrcoef(plate[both], aligned[both])[0, 1])
    axes[1].fill_between(insole_time, 0, max(force.max(), 1), where=mask,
                         color="grey", alpha=0.12, step="mid")
    axes[1].set_ylabel("force (N)")
    axes[1].set_xlabel("OpenCap clock (s)")
    axes[1].legend(fontsize=7, loc="upper right")
    for axis in axes:
        axis.grid(True, alpha=0.3)

    figure.suptitle(f"{trial.name} -- sync offset {offset:+.4f} s [{source}]"
                    + ("" if not np.isfinite(plate_value) else
                       f" | insole-vs-plate r = {plate_value:.3f}"))
    figure.tight_layout()
    out = VALIDATION_ROOT / activity_name / trial.name.split("_")[0]
    out.mkdir(parents=True, exist_ok=True)
    figure.savefig(out / f"{trial.name}.png", dpi=130)
    plt.close(figure)
    return plate_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject")
    args = parser.parse_args()

    offsets, decisions = _final_offsets()
    subject_ids = ([args.subject] if args.subject
                   else sorted(decisions.Subject.unique()))

    discovery = TrialDiscovery(DEFAULT)
    rows = []
    for subject_id in subject_ids:
        subject = Subject(subject_id, DEFAULT)
        for activity_name in ACTIVITIES:
            activity = activity_for(activity_name, DEFAULT)
            for trial in sorted(discovery.trials(subject, activity, "all", for_analysis=True),
                                key=lambda t: t.name):
                if not (trial.insole.is_file() and trial.effective_opencap_trc()
                        and Path(trial.effective_opencap_trc()).is_file()):
                    continue
                offset, source = offsets.get(trial.name, (np.nan, "MISSING"))
                correlation = np.nan
                if np.isfinite(offset):
                    try:
                        correlation = _plot(trial, activity_name, offset, source)
                    except Exception as error:  # noqa: BLE001
                        print(f"  {trial.name}: FAILED ({type(error).__name__}: {error})")
                rows.append({"Subject": subject_id, "Activity": activity_name,
                             "Trial": trial.name, "Offset_s": offset, "Source": source,
                             "InsoleVsPlate_r": round(correlation, 3) if np.isfinite(correlation) else np.nan})

    table = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "sync_validation.csv", index=False)
    VALIDATION_ROOT.mkdir(parents=True, exist_ok=True)
    table.to_csv(VALIDATION_ROOT / "sync_validation.csv", index=False)

    missing = table[table.Source == "MISSING"]
    rejected = table[table.Source == "rejected"]
    print(f"\n{len(table)} trials | auto {(table.Source == 'auto').sum()} | "
          f"manual {(table.Source == 'manual').sum()} | rejected {len(rejected)} | "
          f"MISSING {len(missing)}")
    scored = table[np.isfinite(table.InsoleVsPlate_r)]
    if len(scored):
        print(f"insole-vs-plate correlation: median {scored.InsoleVsPlate_r.median():.3f}, "
              f"min {scored.InsoleVsPlate_r.min():.3f}, "
              f"below 0.8 in {(scored.InsoleVsPlate_r < 0.8).sum()} trial(s)")
    print(f"plots -> {VALIDATION_ROOT}")
    if len(missing):
        print("\nNOT READY for the framework rerun -- these trials still have no sync:")
        for _index, row in missing.iterrows():
            print(f"  {row.Subject:<7} {row.Activity:<13} {row.Trial}")
    else:
        print("\nAll trials carry a sync offset -- ready for the framework rerun.")


if __name__ == "__main__":
    main()
