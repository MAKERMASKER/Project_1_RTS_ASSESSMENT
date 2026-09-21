"""Preflight: one read-only report of every trial/subject's usable status.

Runs before any expensive OpenSim call. Never raises for a data problem --
every subject and every candidate trial gets exactly one status row, so a
missing file, an ambiguous OpenCap session, or an unregistered subject shows
up as a line in a CSV instead of a stack trace three stages into a run.
"""
from pathlib import Path
import pandas as pd

import discovery
from config import DEFAULT
from isokinetic import IsokineticTable
from subject import Subject, normalize_subject_id


class Preflight:
    def __init__(self, config=DEFAULT):
        self.config = config
        self.discovery = discovery.TrialDiscovery(config)

    def known_subject_ids(self) -> list[str]:
        """Union of every subject named in metadata and every data/ folder."""
        ids = set()
        if self.config.metadata_file.is_file():
            table = pd.read_csv(self.config.metadata_file)
            for raw in table["SubjectID"].astype(str):
                if raw.strip():
                    ids.add(normalize_subject_id(raw))
        if self.config.data_root.is_dir():
            for entry in self.config.data_root.iterdir():
                if entry.is_dir():
                    try:
                        ids.add(normalize_subject_id(entry.name))
                    except ValueError:
                        continue
        return sorted(ids)

    def run(self, activity, subject_ids=None) -> pd.DataFrame:
        subject_ids = subject_ids or self.known_subject_ids()
        isokinetic_ids = IsokineticTable(self.config).subject_ids()
        rows = []

        for raw_id in subject_ids:
            subject = Subject(raw_id, self.config)
            if not subject.has_data_folder:
                rows.append(self._row(subject.id, "-", "-", discovery.NO_DATA_FOLDER))
                continue
            if not subject.has_metadata:
                rows.append(self._row(subject.id, "-", "-", discovery.NO_METADATA))
                continue
            if subject.id not in isokinetic_ids:
                rows.append(self._row(subject.id, "-", "-", "NO_ISOKINETIC_ROW"))

            for condition in self.config.conditions:
                for name in self.discovery.candidate_names(subject, activity, condition):
                    rows.append(self._trial_row(subject, activity, condition, name))

        report = pd.DataFrame(rows, columns=["subject", "condition", "trial", "status"])
        self.config.preflight_root.mkdir(parents=True, exist_ok=True)
        out = self.config.preflight_root / f"{activity.name}_trial_report.csv"
        report.to_csv(out, index=False)
        self._print_summary(activity, report, out)
        return report

    def _trial_row(self, subject, activity, condition, name):
        status = self.discovery.classify(subject, activity, name)
        if status == discovery.OK:
            trial = self.discovery._make_trial(subject, activity, name, condition)
            if not trial.insole.is_file():
                status = discovery.MISSING_MATCHING_INSOLE_FILE
            elif not trial.opencap_trc or not trial.opencap_trc.is_file():
                status = discovery.MISSING_OPENCAP_TRC
            elif not trial.opencap_model or not Path(trial.opencap_model).is_file():
                status = discovery.MISSING_SCALED_MODEL
        return self._row(subject.id, condition, name, status)

    @staticmethod
    def _row(subject_id, condition, trial, status):
        return {"subject": subject_id, "condition": condition, "trial": trial, "status": status}

    @staticmethod
    def _print_summary(activity, report: pd.DataFrame, out_path: Path):
        counts = report["status"].value_counts()
        print(f"Preflight | {activity.name} | {len(report)} rows -> {out_path}")
        for status, count in counts.items():
            print(f"  {status}: {count}")
