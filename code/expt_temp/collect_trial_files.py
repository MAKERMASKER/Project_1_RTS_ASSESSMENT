"""collect_trial_files.py -- for one (or more) trials, gather every file
needed to preview it (e.g. in OpenSim Creator's "Preview Experimental Data")
into validation/<SUBJECT>/<ACTIVITY>/trials/<TRIAL_NAME>/, so you don't have
to hunt through results/<SUBJECT>/marker_based/..., results/<SUBJECT>/
framework/..., and data/<SUBJECT>/... by hand.

Layout -- validation/<SUBJECT>/scaled_models/ holds the two genuinely
INDEPENDENT models each pipeline actually computes against (this project
deliberately does NOT share one model between the two pipelines -- see
marker_processor.py's and framework_processor.py's module docstrings),
copied once per subject/condition, not duplicated per trial:
    scaled_models/marker_based/<SUBJECT>_shoe_size_<N>_<CONDITION>_scaled.osim
        The marker-based (ground-truth) pipeline's OWN model: LaiUhlrich2022
        with the study's Biomech57 markerset, scaled from this subject's own
        NOEXO static trial (EXO reuses NOEXO's scale factors and only
        reruns MarkerPlacer) -- see ModelScaler.scale(). This is the exact
        model marker IK/ID actually ran against.
    scaled_models/framework_based/<SUBJECT>_<CONDITION>_opencap_<...>.osim
        OpenCap's OWN scaled model for that condition's session, exactly as
        OpenCap produced it -- untouched, never re-derived. This is the
        exact model framework IK/ID actually ran against
        (trial.opencap_model). Only exists for activities/conditions with
        real OpenCap coverage.
    <ACTIVITY>/trials/<TRIAL_NAME>/
        That trial's own marker-side and framework-side data files.
        triple_hop currently has NO framework processing yet (see
        activities.py's TripleHop -- synchronize() raises
        NotImplementedError, marker-based only so far) -- its
        framework_*/trial_data.npz entries will simply show up as "not
        available" in info.txt, same as any other genuinely-missing file.
        That's expected, not a bug. skater_hop/lateral_hop now have full
        framework processing (manual-sync-based; see select_skh_sync.py/
        select_lh_sync.py in expt_temp).

Usage (from code/, in the opensimPy environment):
    python expt_temp/collect_trial_files.py AB08_NOEXO_SLJ_L_1
    python expt_temp/collect_trial_files.py --subject AB03
    python expt_temp/collect_trial_files.py --subject AB03 --activity skater_hop
    python expt_temp/collect_trial_files.py --all
        (every subject, every activity in ACTIVITY_NAMES)

Missing files (e.g. a trial with no framework output) are noted in each
trial's info.txt rather than causing a failure -- collect what exists.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from activities import activity_for  # noqa: E402
from config import DEFAULT  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from io_tables import read_insole, read_storage, write_storage  # noqa: E402
from subject import Subject  # noqa: E402

SUBJECT_IDS = ("AB03", "AB05", "AB07", "AB08", "AB10", "NTS01")
ACTIVITY_NAMES = ("single_leg_jump", "skater_hop", "lateral_hop", "triple_hop")
VALIDATION_ROOT = CODE_ROOT.parent / "validation"


def mocap_model_path_for(subject, condition: str) -> Path:
    """The marker-based pipeline's OWN Biomech57 anatomically-scaled model
    (ModelScaler.scale()'s output) -- named per its actual on-disk convention
    (see scale_ik_id.py's ModelScaler.scale()). Not activity-specific: one
    model per subject/condition, shared by every activity's marker IK/ID."""
    return (
        DEFAULT.results_root / subject.id / "marker_based" / "models"
        / f"{subject.id}_shoe_size_{subject.shoe_size}_{condition}_scaled.osim"
    )


def marker_root_for(trial) -> Path:
    return DEFAULT.results_root / trial.subject / "marker_based" / trial.condition


def _sources(trial) -> dict:
    """{destination filename: source path} -- every file this trial's
    preview could use. Not every entry will exist for every trial (e.g. a
    trial with no OpenCap recording, or an activity with no framework
    processing at all yet, has no framework files)."""
    marker_root = marker_root_for(trial)
    return {
        "insole_raw.txt": trial.insole,
        "marker_static_or_dynamic.trc": trial.marker_trc,
        "marker_force_plate.mot": trial.force_mot,
        "marker_ik.mot": marker_root / "ik" / f"{trial.name}_ik.mot",
        "marker_inverse_dynamics.sto": marker_root / "id" / f"{trial.name}_inverse_dynamics.sto",
        "marker_external_loads.xml": marker_root / "id" / f"setup_{trial.name}" / "ExternalLoads.xml",
        "framework_opencap.trc": trial.effective_opencap_trc(),
        "framework_ik.mot": trial.effective_opencap_ik(),
        "framework_grf.mot": trial.output / f"{trial.name}_opencap_insole_grf.mot",
        "framework_inverse_dynamics.sto": trial.output / f"{trial.name}_opencap_insole_id.sto",
        "framework_external_loads.xml": trial.output / "setup" / "ExternalLoads.xml",
        "trial_data.npz": trial.output / "trial_data.npz",
    }


def ensure_subject_models(subject, trial) -> dict:
    """Copy this subject/condition's two INDEPENDENT models into
    validation/<SUBJECT>/scaled_models/{marker_based,framework_based}/
    (once each -- every trial of that condition/activity reuses the same
    files, so this is idempotent and cheap to call per trial). Returns
    {"marker": path_or_none, "framework": path_or_none}."""
    result = {}

    mocap_source = mocap_model_path_for(subject, trial.condition)
    marker_dir = VALIDATION_ROOT / subject.id / "scaled_models" / "marker_based"
    if mocap_source.is_file():
        marker_dir.mkdir(parents=True, exist_ok=True)
        mocap_destination = marker_dir / mocap_source.name
        shutil.copy2(mocap_source, mocap_destination)
        result["marker"] = mocap_destination
    else:
        result["marker"] = None

    opencap_source = Path(trial.opencap_model) if trial.opencap_model else None
    framework_dir = VALIDATION_ROOT / subject.id / "scaled_models" / "framework_based"
    if opencap_source and opencap_source.is_file():
        framework_dir.mkdir(parents=True, exist_ok=True)
        opencap_destination = framework_dir / f"{subject.id}_{trial.condition}_opencap_{opencap_source.name}"
        shutil.copy2(opencap_source, opencap_destination)
        result["framework"] = opencap_destination
    else:
        result["framework"] = None
    return result


def _insole_shift(trial):
    """The framework pipeline's own insole<->OpenCap shift (activity.
    synchronize()'s shift_s, already baked into framework_grf.mot/
    framework_inverse_dynamics.sto) -- None if this trial has no framework
    processing (e.g. triple_hop, or a skipped trial)."""
    package = trial.output / "trial_data.npz"
    if not package.is_file():
        return None
    data = np.load(package, allow_pickle=True)
    return float(data["shift_s"])


def _marker_shift(trial):
    """The marker-based (ground-truth) pipeline's own clock has no defined
    relationship to the framework clock in production code -- the two are
    genuinely independent recording systems. outcomes.py's analyze_trial()
    computes an ad hoc alignment for its own RMS comparisons only (matching
    the framework's synced landing-force peak against the marker force
    plate's own peak) and never persists it; this reuses that exact same
    heuristic so validation files can be visually overlaid too. None if
    there's no framework trial_data.npz or no marker force MOT to align
    against.

    Side resolution: `trial.side` for every unilateral activity (SLJ, LH);
    for a bilateral activity with no committed side (SkaterHop,
    trial.side is None), the npz's own "force"/"grf_N" is already the
    fixed sync-anchor side ("R" -- see SkaterHop.synchronize()), so the
    marker force plate to match against must be plate 1 (right) too --
    falling through to trial.side == "R" (False for None) would silently
    pick the WRONG (left) plate for SKH."""
    package = trial.output / "trial_data.npz"
    if not package.is_file() or not trial.force_mot.is_file():
        return None
    data = np.load(package, allow_pickle=True)
    time = data["time"]
    force = data["force"]
    side = trial.side if trial.side is not None else "R"
    plate = 1 if side == "R" else 2
    force_plate = read_storage(trial.force_mot)
    marker_vertical = np.abs(force_plate[f"ground_force_{plate}_vy"].to_numpy(float))
    return float(time[np.argmax(force)] - force_plate.time.iloc[np.argmax(marker_vertical)])


def _write_shifted_storage(source_path: Path, shift: float, destination: Path):
    table = read_storage(source_path)
    table = table.assign(time=table["time"] + shift)
    write_storage(destination, table)


def _write_shifted_insole(trial, shift: float, destination: Path):
    table = read_insole(trial.insole)
    table = table.assign(time=table["time"] + shift)
    write_storage(destination, table, title="insole_synced")


def collect(trial, activity) -> Path:
    destination = VALIDATION_ROOT / trial.subject / activity.name / "trials" / trial.name
    destination.mkdir(parents=True, exist_ok=True)
    models = ensure_subject_models(Subject(trial.subject, DEFAULT), trial)

    collected, missing = [], []
    for filename, source in _sources(trial).items():
        source = Path(source) if source else None
        if source is not None and source.is_file():
            shutil.copy2(source, destination / filename)
            collected.append(filename)
        else:
            missing.append(filename)

    # Everything on one common (synced) clock: the framework files
    # (framework_opencap.trc/framework_ik.mot/framework_grf.mot/
    # framework_inverse_dynamics.sto) are already on OpenCap's clock, which
    # insole gets shifted onto (activity.synchronize()'s shift_s). Add
    # shifted companions for insole_raw.txt (onto that same framework
    # clock) and for the marker-based files (onto the framework clock too,
    # via outcomes.py's own peak-matching heuristic -- see _marker_shift())
    # so every collected file overlays on one timeline. None of this
    # exists yet for an activity with no framework processing (triple_hop).
    synced_lines = []
    insole_shift = _insole_shift(trial)
    if insole_shift is not None:
        _write_shifted_insole(trial, insole_shift, destination / "insole_synced.mot")
        collected.append("insole_synced.mot")
        synced_lines.append(f"  insole_synced.mot: shift={insole_shift:+.3f}s (activity.synchronize()'s shift_s)")

    marker_shift = _marker_shift(trial)
    if marker_shift is not None:
        marker_root = marker_root_for(trial)
        pairs = [
            (marker_root / "ik" / f"{trial.name}_ik.mot", "marker_ik_synced.mot"),
            (marker_root / "id" / f"{trial.name}_inverse_dynamics.sto", "marker_inverse_dynamics_synced.sto"),
            (trial.force_mot, "marker_force_plate_synced.mot"),
        ]
        for source_path, out_name in pairs:
            if source_path.is_file():
                _write_shifted_storage(source_path, marker_shift, destination / out_name)
                collected.append(out_name)
        synced_lines.append(
            f"  marker_*_synced files: shift={marker_shift:+.3f}s "
            f"(outcomes.py's force-peak-matching heuristic, marker clock -> framework clock)"
        )

    def _model_line(label, path):
        if path is None:
            return f"  {label}: MISSING"
        return f"  {label}: ../../../scaled_models/{path.parent.name}/{path.name}"

    info_lines = [
        f"Trial: {trial.name}  Activity: {activity.name}",
        f"Subject: {trial.subject}  Condition: {trial.condition}  Stance side: {trial.side}",
        "",
        "Models (in validation/<SUBJECT>/scaled_models/, not duplicated per trial --",
        "these are two INDEPENDENTLY scaled models, one per pipeline, never shared):",
        _model_line("Marker-based (ModelScaler.scale())", models["marker"]),
        _model_line("Framework (OpenCap's own scaled model)", models["framework"]),
        "",
        "Collected files:",
        *[f"  {name}" for name in collected],
    ]
    if missing:
        info_lines += ["", "Not available for this trial (skipped):", *[f"  {name}" for name in missing]]
    if synced_lines:
        info_lines += [
            "",
            "Time-synced companions (everything else above is on its OWN pipeline's",
            "native clock -- use the _synced files to overlay signals in one viewer):",
            *synced_lines,
        ]
    else:
        info_lines += [
            "",
            "No time-synced companions -- this activity has no framework processing yet"
            " (e.g. triple_hop), so there's no framework clock to align onto.",
        ]
    (destination / "info.txt").write_text("\n".join(info_lines) + "\n")
    return destination


def _trials_for(subject_id, activity_name):
    discovery = TrialDiscovery(DEFAULT)
    activity = activity_for(activity_name, DEFAULT)
    subject = Subject(subject_id, DEFAULT)
    return activity, discovery.trials(subject, activity, "all", for_analysis=True)


def _all_trials(activity_names=ACTIVITY_NAMES, subject_ids=SUBJECT_IDS):
    result = []
    for activity_name in activity_names:
        for subject_id in subject_ids:
            activity, trials = _trials_for(subject_id, activity_name)
            result.extend((trial, activity) for trial in trials)
    return result


def _find_trial(trial_name: str):
    subject_id = trial_name.split("_")[0]
    for activity_name in ACTIVITY_NAMES:
        activity, trials = _trials_for(subject_id, activity_name)
        for trial in trials:
            if trial.name == trial_name:
                return trial, activity
    raise ValueError(f"Trial not found in any of {ACTIVITY_NAMES}: {trial_name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trial", nargs="?", help="One trial name, e.g. AB08_NOEXO_SLJ_L_1")
    parser.add_argument("--subject", help="Collect every analyzable trial for this subject")
    parser.add_argument("--activity", choices=ACTIVITY_NAMES, help="Restrict to one activity (default: all of ACTIVITY_NAMES)")
    parser.add_argument("--all", action="store_true", help="Collect every analyzable trial for all subjects/activities")
    args = parser.parse_args()

    if args.all:
        pairs = _all_trials()
    elif args.subject:
        activity_names = (args.activity,) if args.activity else ACTIVITY_NAMES
        pairs = _all_trials(activity_names=activity_names, subject_ids=(args.subject,))
    elif args.trial:
        pairs = [_find_trial(args.trial)]
    else:
        parser.error("Provide a trial name, --subject, or --all")

    for trial, activity in pairs:
        destination = collect(trial, activity)
        print(f"{activity.name}/{trial.name} -> {destination}")


if __name__ == "__main__":
    main()
