"""TrialDiscovery: find candidate trials and classify why each is or isn't usable.

Two entry points:
  - candidate_names(...): every raw C3D name the glob matches, unfiltered.
  - classify(...): the preflight status for one candidate name.
  - trials(...): only the clean, non-excluded names, built into Trial objects
    -- what the real processing pipeline consumes.

Nothing here raises for a data problem. A subject with no data folder, a
trial with no matching insole file, an ambiguous OpenCap session -- all of
these are reported as a status string, not an exception. That is what makes
Preflight possible without a second parallel implementation.
"""
from pathlib import Path
import re

import pandas as pd

from config import DEFAULT
from subject import Subject, Trial

# Status constants used by both TrialDiscovery and Preflight.
OK = "OK"
NO_DATA_FOLDER = "NO_DATA_FOLDER"
NO_METADATA = "NO_METADATA"
DISCARD_OR_OLD = "DISCARD_OR_OLD"
LEGACY_SUFFIX = "LEGACY_SUFFIX"
UNRECOGNIZED_SUFFIX = "UNRECOGNIZED_SUFFIX"
USER_EXCLUDED_RULE = "USER_EXCLUDED_RULE"
MISSING_MATCHING_INSOLE_FILE = "MISSING_MATCHING_INSOLE_FILE"
MISSING_OPENCAP_TRC = "MISSING_OPENCAP_TRC"
MISSING_OPENCAP_SESSION = "MISSING_OPENCAP_SESSION"
MISSING_SCALED_MODEL = "MISSING_SCALED_MODEL"


def _excluded_trial_status(name: str, reason: str) -> str:
    return f"EXCLUDED_TRIAL_STATUS({reason})"


class TrialDiscovery:
    def __init__(self, config=DEFAULT):
        self.config = config

    # ---- raw candidates -------------------------------------------------

    def candidate_names(self, subject: Subject, activity, condition: str) -> list[str]:
        if not subject.has_data_folder:
            return []
        raw = subject.data_folder / "RAW" / "MOCAP"
        if not raw.is_dir():
            return []
        pattern = f"{subject.id}_{condition.upper()}_{activity.token}_*.c3d"
        return sorted({path.stem for path in raw.glob(pattern)})

    def classify(self, subject: Subject, activity, name: str, for_analysis: bool = False) -> str:
        """Preflight status for one candidate C3D stem. Never raises."""
        lowered = name.lower()
        if "discard" in lowered or "_old" in lowered:
            return DISCARD_OR_OLD
        if re.search(r"_\d{3}$", name):
            return LEGACY_SUFFIX
        side_group = "_[LR]" if activity.has_side_token else ""
        expected = rf"^{re.escape(subject.id)}_[A-Z]+_{re.escape(activity.token)}{side_group}_\d+$"
        if not re.match(expected, name.upper()):
            return UNRECOGNIZED_SUFFIX

        if name.lower() in {excluded.lower() for excluded in subject.rules["exclude"]}:
            return USER_EXCLUDED_RULE

        status_row = self._trial_status_row(subject, name)
        if status_row is not None:
            column = "Analysis" if for_analysis else "Process"
            value = str(status_row[column]).strip().lower()
            if value in ("false", "0", "no"):
                return _excluded_trial_status(name, str(status_row["Reason"]))
        return OK

    def _trial_status_row(self, subject: Subject, name: str):
        path = self.config.trial_status_file
        if not path.is_file():
            return None
        table = pd.read_csv(path)
        rows = table[
            table["Subject"].astype(str).str.upper().eq(subject.id)
            & table["Trial"].astype(str).str.lower().eq(name.lower())
        ]
        return rows.iloc[0] if len(rows) else None

    # ---- resolved trials --------------------------------------------------

    def trials(self, subject: Subject, activity, condition: str = "all", for_analysis: bool = False) -> list[Trial]:
        conditions = self.config.conditions if condition.lower() == "all" else (condition.upper(),)
        result = []
        for one_condition in conditions:
            for name in self.candidate_names(subject, activity, one_condition):
                if self.classify(subject, activity, name, for_analysis) == OK:
                    result.append(self._make_trial(subject, activity, name, one_condition))
        return result

    def _make_trial(self, subject: Subject, activity, name: str, condition: str) -> Trial:
        base = subject.data_folder
        rules = subject.rules
        side = subject.side_from_name(name, required=activity.has_side_token)

        insole_name = rules["insole_aliases"].get(name, name)
        opencap_name = rules["opencap_aliases"].get(name, name)

        opencap_root, opencap_model = self._resolve_opencap(subject, activity, name, condition, rules)

        output = self.config.results_root / subject.id / "framework" / activity.name / condition.lower() / name
        repaired = None
        repaired_ik = None
        if name in rules["repaired_trials"]:
            candidate = base / "PROCESSED" / "OPENCAP_REPAIRED" / f"{opencap_name}.trc"
            if candidate.is_file():
                repaired = candidate
                # May not exist yet -- FrameworkProcessor regenerates it from
                # the repaired TRC on demand (see Trial.effective_opencap_ik).
                repaired_ik = base / "PROCESSED" / "OPENCAP_REPAIRED" / f"{opencap_name}_ik.mot"

        return Trial(
            subject=subject.id,
            condition=condition,
            activity=activity.name,
            name=name,
            side=side,
            insole=base / "RAW" / "INSOLE" / f"{insole_name}.txt",
            c3d=base / "RAW" / "MOCAP" / f"{name}.c3d",
            marker_trc=base / "PROCESSED" / "MOCAP" / "marker_data" / f"{name}_markers_PROCESSED.trc",
            force_mot=base / "PROCESSED" / "MOCAP" / "force_data" / f"{name}_forces_PROCESSED.mot",
            opencap_trc=(opencap_root / "MarkerData" / f"{opencap_name}.trc") if opencap_root else Path(""),
            opencap_ik=(opencap_root / "OpenSimData" / "Kinematics" / f"{opencap_name}.mot") if opencap_root else Path(""),
            opencap_model=opencap_model or Path(""),
            output=output,
            repaired_opencap_trc=repaired,
            repaired_opencap_ik=repaired_ik,
        )

    def _resolve_opencap(self, subject: Subject, activity, name: str, condition: str, rules: dict):
        """Return (session_folder, scaled_model_path). Either may be None."""
        base = subject.data_folder
        override = rules["opencap_session_overrides"].get(name)
        folder_name = override or self._session_folder_name(subject, activity, condition)
        if folder_name is None:
            return None, None

        for stage in ("PROCESSED", "RAW"):
            candidate = base / stage / "OPENCAP" / folder_name
            if candidate.is_dir():
                models = list((candidate / "OpenSimData" / "Model").glob("*_scaled.osim"))
                model = models[0] if len(models) == 1 else None
                return candidate, model
        return None, None

    def _session_folder_name(self, subject: Subject, activity, condition: str):
        path = self.config.session_file
        if not path.is_file():
            return None
        manifest = pd.read_csv(path, dtype=str)
        rows = manifest["Condition"].astype(str).str.upper().eq(condition.upper())
        rows &= (
            manifest["Activities"].astype(str).str.lower().eq("all")
            | manifest["Activities"].astype(str).str.contains(activity.name, case=False, regex=False)
        )
        if "SubjectID" in manifest:
            rows &= manifest["SubjectID"].astype(str).str.upper().eq(subject.id)
        matched = manifest.loc[rows]
        if len(matched) != 1:
            return None
        folder = matched["Folder"].iloc[0]
        if pd.notna(folder) and str(folder).strip():
            return str(folder).strip()
        # Folder is populated after a real download/verification pass (see
        # opencap_admin.download.OpenCapDownloader), but the folder name is
        # always deterministic from SessionID -- fall back to that instead
        # of treating a blank Folder as "session doesn't exist yet."
        session_id = matched["SessionID"].iloc[0]
        if pd.notna(session_id) and str(session_id).strip():
            return f"OpenCapData_{str(session_id).strip()}"
        return None
