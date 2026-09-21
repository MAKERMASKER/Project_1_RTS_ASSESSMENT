"""Command-line entry point.

download | preflight | marker | calibrate | framework | analyze | correlate
| repair-filter | repair-validate | repair-promote

The repair-* commands are three explicit, separate steps -- see
trc_repair.py's module docstring. IK is never rerun for a trial until
repair-promote has been run for it; `framework`/`analyze` only pick up a
repair automatically once it's promoted.
"""
import argparse

from activities import activity_for
from correlation import CorrelationAnalysis
from discovery import TrialDiscovery
from opencap_admin.download import OpenCapDownloader
from preflight import Preflight
from project import RtsAssessmentProject
from subject import Subject
from trc_repair import AutomaticTrcFilter, RepairPromoter, TrcRepairValidator


def _find_trial(subject, activity, trial_name):
    discovery = TrialDiscovery()
    for condition in ("NOEXO", "EXO"):
        for trial in discovery.trials(subject, activity, condition):
            if trial.name.lower() == trial_name.lower():
                return trial
    raise ValueError(f"No accepted trial named {trial_name!r} for {subject.id}")


def main():
    parser = argparse.ArgumentParser(description="RTS assessment SLJ pipeline")
    parser.add_argument(
        "command",
        choices=[
            "download", "preflight", "marker", "calibrate", "framework", "analyze", "correlate",
            "repair-filter", "repair-validate", "repair-promote",
        ],
    )
    parser.add_argument("--subject", help="Subject ID, e.g. AB08 (omit for preflight/correlate across all subjects)")
    parser.add_argument("--activity", default="single_leg_jump")
    parser.add_argument("--condition", default="all")
    parser.add_argument("--trial", default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--skip-c3d", action="store_true", help="Reuse existing PROCESSED marker/force files")
    parser.add_argument("--credentials", default=None, help="Path to OpenCap credentials file (download only)")
    args = parser.parse_args()

    if args.command == "download":
        if not args.subject:
            parser.error("--subject is required for 'download'")
        subject = Subject(args.subject)
        folders = OpenCapDownloader(credentials_path=args.credentials).download_subject(subject)
        print(f"Downloaded {len(folders)} session folder(s): {', '.join(folders)}")
        return

    if args.command == "preflight":
        activity = activity_for(args.activity)
        subject_ids = [args.subject] if args.subject else None
        Preflight().run(activity, subject_ids=subject_ids)
        return

    if args.command == "correlate":
        activity = activity_for(args.activity)
        if args.subject:
            subjects = [Subject(args.subject)]
        else:
            subjects = [Subject(subject_id) for subject_id in Preflight().known_subject_ids()]
        analysis = CorrelationAnalysis()
        joined, statistics, _agreement = analysis.run(subjects, activity, make_plots=not args.no_plots)
        print(f"{len(joined)} joined subject/comparison/metric rows -> results/_correlation/")
        print(statistics.to_string(index=False))
        return

    if args.command in ("repair-filter", "repair-validate", "repair-promote"):
        if not (args.subject and args.trial):
            parser.error(f"--subject and --trial are required for '{args.command}'")
        subject = Subject(args.subject)
        activity = activity_for(args.activity)
        trial = _find_trial(subject, activity, args.trial)
        repair_dir = trial.output / "trc_repair"
        filtered_trc = repair_dir / "filtered_10hz.trc"

        if args.command == "repair-filter":
            result = AutomaticTrcFilter().process_trial(trial, repair_dir)
            print(f"Filtered TRC:  {result['filtered_trc']}")
            print(f"Quality report: {repair_dir / 'marker_quality.csv'}")
            print(f"Markers flagged for review: {result['markers_flagged_for_review']}")
            print("Review the quality report, then run 'repair-validate' before promoting.")
            return

        if args.command == "repair-validate":
            if not filtered_trc.is_file():
                parser.error(f"No filtered TRC at {filtered_trc} -- run 'repair-filter' first")
            summary = TrcRepairValidator().validate(subject, activity, trial, filtered_trc)
            print(summary.to_string(index=False))
            print(f"Comparison plot: {trial.output / 'trc_repair_validation' / 'repaired_vs_baseline_vs_marker.png'}")
            print("If this looks like a real improvement, run 'repair-promote' to confirm it.")
            return

        if args.command == "repair-promote":
            if not filtered_trc.is_file():
                parser.error(f"No filtered TRC at {filtered_trc} -- run 'repair-filter' (and ideally 'repair-validate') first")
            destination = RepairPromoter().promote(args.subject, trial, filtered_trc)
            print(f"Promoted: {destination}")
            print(f"subject_rules/{args.subject}.json now lists {trial.name!r} in repaired_trials.")
            print("The next 'framework' run for this trial will rerun IK from the repaired TRC automatically.")
            return

    if not args.subject:
        parser.error(f"--subject is required for '{args.command}'")

    project = RtsAssessmentProject(args.subject)
    if args.command == "marker":
        project.run_marker(args.activity, args.condition, convert_c3d=not args.skip_c3d)
    elif args.command == "calibrate":
        path = project.run_calibration()
        print(f"Calibration written to {path}")
    elif args.command == "framework":
        project.run_framework(args.activity, args.condition, args.trial)
        project.run_analysis(args.activity, make_plots=not args.no_plots)
    elif args.command == "analyze":
        ratios = project.run_analysis(args.activity, make_plots=not args.no_plots)
        print(ratios.to_string(index=False))


if __name__ == "__main__":
    main()
