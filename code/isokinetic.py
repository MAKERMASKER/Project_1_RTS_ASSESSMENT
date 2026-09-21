"""IsokineticTable: load, normalize, and reshape the gold-standard isokinetic results.

Trusts the sheet's own LSI columns (PEAK_TORQUE_NOEXO_LSI, WORK_WITHEXO_LSI,
...) as the correlation target rather than recomputing LSI from raw
torque/work -- that sheet's formula already matches this project's own LSI
direction exactly (verified: 100 * LEFT_NOEXO / RIGHT_NOEXO, etc. -- see
CLAUDE.md's Domain facts), so recomputing would just be re-deriving the same
number with more steps and more chances to get the direction wrong.
"""
import pandas as pd

from config import DEFAULT
from subject import normalize_subject_id

# metric -> {movement-task-style condition label -> isokinetic sheet column}.
# The condition labels here ("NOEXO", "WITHEXO") match config.isokinetic_condition_map's
# *values* -- i.e. what a movement-task "NOEXO"/"EXO" condition maps onto.
LONG_METRIC_COLUMNS = {
    "peak_torque": {"NOEXO": "PEAK_TORQUE_NOEXO_LSI", "WITHEXO": "PEAK_TORQUE_WITHEXO_LSI"},
    "total_work": {"NOEXO": "WORK_NOEXO_LSI", "WITHEXO": "WORK_WITHEXO_LSI"},
}


class IsokineticTable:
    def __init__(self, config=DEFAULT):
        self.config = config

    def load_wide(self) -> pd.DataFrame:
        """The sheet as-is, with its subject column renamed to SubjectID and normalized."""
        table = pd.read_excel(self.config.isokinetic_file, engine="openpyxl")
        subject_column = next((c for c in table.columns if "SUBJECT" in c.upper()), None)
        if subject_column is None:
            raise ValueError(f"No subject-identifier column found in {self.config.isokinetic_file}")
        table = table.rename(columns={subject_column: "SubjectID"})
        table = table[table["SubjectID"].notna()].copy()
        table = table[table["SubjectID"].astype(str).str.strip() != ""]
        table["SubjectID"] = table["SubjectID"].astype(str).map(normalize_subject_id)
        return table

    def load_long(self) -> pd.DataFrame:
        """One row per (subject, condition, metric) with the sheet's own LSI value."""
        wide = self.load_wide()
        rows = []
        for metric, condition_columns in LONG_METRIC_COLUMNS.items():
            for condition, column in condition_columns.items():
                if column not in wide.columns:
                    continue
                for _, row in wide.iterrows():
                    value = row[column]
                    if pd.notna(value):
                        rows.append({
                            "subject": row["SubjectID"],
                            "condition": condition,
                            "metric": metric,
                            "iso_lsi": float(value),
                        })
        return pd.DataFrame(rows, columns=["subject", "condition", "metric", "iso_lsi"])

    def subject_ids(self) -> set:
        if not self.config.isokinetic_file.is_file():
            return set()
        return set(self.load_wide()["SubjectID"])
