"""IsokineticTable against the real Isokinetic_Data.xlsx: this doubles as
living documentation of the current subject overlap -- it should start
passing more subjects as more isokinetic rows and data/ folders arrive,
never fewer."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DEFAULT
from isokinetic import IsokineticTable

pytestmark = pytest.mark.skipif(
    not DEFAULT.isokinetic_file.is_file(), reason="Isokinetic_Data.xlsx not present"
)


def test_subject_ids_are_normalized():
    ids = IsokineticTable(DEFAULT).subject_ids()
    assert "AB08" in ids
    assert "AB9" not in ids  # the sheet's unpadded form must not survive normalization
    assert "AB09" in ids


def test_known_overlap_with_data_folders():
    iso_ids = IsokineticTable(DEFAULT).subject_ids()
    data_ids = {path.name for path in DEFAULT.data_root.iterdir() if path.is_dir()}
    assert (iso_ids & data_ids) == {"AB03", "AB05", "AB07", "AB08"}
    assert "NTS01" not in iso_ids  # has movement data but no isokinetic row


def test_long_form_condition_and_metric_values():
    long = IsokineticTable(DEFAULT).load_long()
    assert set(long["condition"]) <= {"NOEXO", "WITHEXO"}
    assert set(long["metric"]) <= {"peak_torque", "total_work"}

    row = long[(long.subject == "AB08") & (long.condition == "NOEXO") & (long.metric == "peak_torque")]
    assert len(row) == 1
    # Verified by hand against the sheet in the project plan: 100 * LEFT_NOEXO / RIGHT_NOEXO for AB08.
    assert row["iso_lsi"].iloc[0] == pytest.approx(97.435897, rel=1e-4)
