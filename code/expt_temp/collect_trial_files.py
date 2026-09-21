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
        skater_hop/lateral_hop currently have NO framework processing yet
        (see activities.py's SkaterHop/LateralHop -- synchronize() raises
        NotImplementedError) -- their framework_*/trial_data.npz entries
        will simply show up as "not available" in info.txt, same as any
        other genuinely-missing file. That's expected, not a bug.

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

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from activities import activity_for  # noqa: E402
from config import DEFAULT  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from subject import Subject  # noqa: E402

SUBJECT_IDS = ("AB03", "AB05", "AB07", "AB08", "AB10", "NTS01")
ACTIVITY_NAMES = ("single_leg_jump", "skater_hop", "lateral_hop")
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
