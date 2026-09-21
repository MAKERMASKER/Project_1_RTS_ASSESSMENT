"""Subject and Trial: identity, metadata, and per-subject rule overrides.

Trial discovery (globbing raw files, resolving OpenCap sessions, applying
trial_status.csv) lives in discovery.py, not here -- this module only knows
"who is this subject and what do we know about them," and never raises for
data that is merely absent (a missing data/<ID> folder, a subject missing
from subject_metadata.csv). Absence is reported by preflight.py, not an
exception raised deep in a pipeline run.
"""
from dataclasses import dataclass, fields
from functools import cached_property
from pathlib import Path
from typing import Optional
import json
import re

import pandas as pd

from config import DEFAULT


def normalize_subject_id(raw: str) -> str:
    """Canonicalize a subject id: uppercase, digits zero-padded to 2 places.

    ``AB5`` -> ``AB05``, ``AB08`` -> ``AB08``, ``nts1`` -> ``NTS01``.
    Applied everywhere a subject id is read from a filename, folder name, or
    any of the metadata CSVs / Isokinetic_Data.xlsx, so every table in the
    codebase keys on the same canonical form.
    """
    raw = raw.strip().upper()
    match = re.fullmatch(r"([A-Z]+)(\d+)", raw)
    if not match:
        raise ValueError(f"Unrecognized subject id: {raw!r}")
    prefix, digits = match.groups()
    return f"{prefix}{int(digits):02d}"


@dataclass(frozen=True)
class Trial:
    """All identity and file paths for one trial. Pure data, no behavior."""

    subject: str
    condition: str
    activity: str
    name: str
    side: Optional[str]
    insole: Path
    c3d: Path
    marker_trc: Path
    force_mot: Path
    opencap_trc: Path
    opencap_ik: Path
    opencap_model: Path
    output: Path
    repaired_opencap_trc: Optional[Path] = None
    repaired_opencap_ik: Optional[Path] = None

    def __getitem__(self, key):
        return getattr(self, key)

    def as_dict(self):
        return {item.name: getattr(self, item.name) for item in fields(self)}

    def effective_opencap_trc(self) -> Path:
        """The TRC FrameworkProcessor should read: repaired if promoted, else raw."""
        if self.repaired_opencap_trc is not None and self.repaired_opencap_trc.is_file():
            return self.repaired_opencap_trc
        return self.opencap_trc

    def effective_opencap_ik(self) -> Path:
        """The IK FrameworkProcessor should read.

        OpenCap's own cloud IK (`opencap_ik`) was computed against the
        ORIGINAL, unrepaired TRC -- once a repair is promoted it is stale.
        This path points at the IK re-derived from the repaired TRC instead
        (see opencap_ik.OpenCapIkRunner); that file may not exist yet on
        disk -- FrameworkProcessor is responsible for generating it there
        before reading it, the same lazy-cache pattern as the shared Lai
        model. Unlike `effective_opencap_trc`, this does not check
        `is_file()`: the repaired TRC being present is what decides whether a
        repair is active, not whether the IK has been (re)computed yet.
        """
        if self.repaired_opencap_trc is not None and self.repaired_opencap_trc.is_file():
            return self.repaired_opencap_ik
        return self.opencap_ik


class Subject:
    """One subject's identity, metadata row, and merged rule overrides."""

    def __init__(self, subject_id: str, config=DEFAULT):
        self.config = config
        self.id = normalize_subject_id(subject_id)

    @property
    def data_folder(self) -> Path:
        return self.config.data_root / self.id

    @property
    def has_data_folder(self) -> bool:
        return self.data_folder.is_dir()

    @cached_property
    def metadata(self):
        """This subject's row from subject_metadata.csv, or None if absent.

        Never raises: subjects registered in the shared metadata but not yet
        collected (e.g. future participants) are a normal, reportable state,
        not an error.
        """
        table = pd.read_csv(self.config.metadata_file)
        ids = table["SubjectID"].astype(str).map(
            lambda raw: normalize_subject_id(raw) if raw.strip() else raw
        )
        rows = table[ids == self.id]
        if len(rows) == 0:
            return None
        if len(rows) > 1:
            raise ValueError(
                f"Expected at most one metadata row for {self.id}; found {len(rows)}"
            )
        return rows.iloc[0]

    @property
    def has_metadata(self) -> bool:
        return self.metadata is not None

    @property
    def shoe_size(self) -> Optional[int]:
        if self.metadata is None or "ShoeSize" not in self.metadata.index:
            return None
        size = int(self.metadata["ShoeSize"])
        if size not in (5, 7):
            raise ValueError(f"{self.id} ShoeSize must be 5 or 7; found {size}")
        return size

    @property
    def mass_kg(self) -> Optional[float]:
        if self.metadata is None or "Mass_kg" not in self.metadata.index:
            return None
        return float(self.metadata["Mass_kg"])

    @property
    def height_m(self) -> Optional[float]:
        if self.metadata is None or "Height_m" not in self.metadata.index:
            return None
        return float(self.metadata["Height_m"])

    @property
    def exo_leg(self) -> Optional[str]:
        if self.metadata is None or "ExoLeg" not in self.metadata.index:
            return None
        return str(self.metadata["ExoLeg"]).upper()

    @cached_property
    def rules(self) -> dict:
        """Merge default.json with an optional <ID>.json override."""
        default_path = self.config.subject_rules_root / "default.json"
        if not default_path.is_file():
            raise FileNotFoundError(default_path)
        rules = json.loads(default_path.read_text())
        subject_path = self.config.subject_rules_root / f"{self.id}.json"
        if subject_path.is_file():
            rules.update(json.loads(subject_path.read_text()))

        rules["exclude"] = set(rules.get("exclude", []))
        rules["use_last_flight_peak"] = set(rules.get("use_last_flight_peak", []))
        rules["repaired_trials"] = set(rules.get("repaired_trials", []))
        rules.setdefault("landing_ordinal", {})
        rules.setdefault("opencap_aliases", {})
        rules.setdefault("insole_aliases", {})
        rules.setdefault("opencap_session_overrides", {})
        rules.setdefault("force_scale_mode", {"L": "none", "R": "none"})
        rules.setdefault("force_scale", {"L": 1.0, "R": 1.0})
        rules.setdefault("insole_vertical_weight", 1.0)
        rules.setdefault("com_vertical_weight", 0.0)
        rules.setdefault("rotation_force_activities", [])
        rules.setdefault("rotation_force_weight_x", 0.0)
        rules.setdefault("rotation_force_weight_z", 0.0)

        for field_name in ("force_scale_mode", "force_scale"):
            value = rules[field_name]
            if not isinstance(value, dict):
                raise ValueError(
                    f"{self.id} subject_rules {field_name!r} must be a per-side "
                    f"dict {{'L': ..., 'R': ...}}, not a single shared value "
                    f"(a single bilateral value was the old code's bug)."
                )
        return rules

    def side_from_name(self, name: str, required: bool = True) -> Optional[str]:
        match = re.search(r"_([LR])(?:_|$)", name.upper())
        if match:
            return match.group(1)
        if not required:
            # A genuinely bilateral activity (e.g. Walk, SkaterHop) whose
            # raw filenames carry no side segment at all -- not an error,
            # just no single side to report (see Activity.has_side_token).
            # SkaterHop trials alternate legs within one trial, so there is
            # no single representative side to fall back to here (a prior
            # version of this method guessed one from exo_leg metadata --
            # removed as factually wrong for an alternating-leg movement).
            return None
        raise ValueError(f"Cannot infer a side from {name}")
