"""TrialDiscovery/Preflight against the real data/ tree: regression test for
the specific data-quality issues this project is built to tolerate (a
missing insole file, subjects with no data/ folder, user-rule exclusions),
so a future change can't silently start crashing or silently start hiding
these again."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import discovery
from activities import activity_for
from config import DEFAULT
from preflight import Preflight

pytestmark = pytest.mark.skipif(not DEFAULT.data_root.is_dir(), reason="data/ not present")


@pytest.fixture(scope="module")
def report():
    return Preflight(DEFAULT).run(activity_for("single_leg_jump"))


def _status_for(report, subject: str, trial: str) -> str:
    rows = report[(report.subject == subject) & (report.trial == trial)]
    assert len(rows) == 1, f"expected exactly one preflight row for {subject}/{trial}"
    return rows["status"].iloc[0]


def test_missing_insole_file_is_flagged_not_silently_dropped(report):
    assert "MISSING_MATCHING_INSOLE_FILE" in _status_for(report, "AB08", "AB08_NOEXO_SLJ_R_4")


@pytest.mark.parametrize("subject", ["TS01", "TS02", "TS03", "NTS03"])
def test_subjects_without_a_data_folder_are_reported_not_raised(report, subject):
    rows = report[report.subject == subject]
    assert len(rows) == 1
    assert rows["status"].iloc[0] == discovery.NO_DATA_FOLDER


def test_ab08_user_excluded_trials_are_flagged(report):
    assert _status_for(report, "AB08", "AB08_NOEXO_SLJ_R_1") == discovery.USER_EXCLUDED_RULE
    assert _status_for(report, "AB08", "AB08_NOEXO_SLJ_R_3") == discovery.USER_EXCLUDED_RULE


def test_nts01_has_no_isokinetic_row(report):
    rows = report[(report.subject == "NTS01") & (report.trial == "-")]
    assert (rows["status"] == "NO_ISOKINETIC_ROW").any()


def test_ab08_noexo_slj_left_trials_are_ok(report):
    for trial_number in range(1, 6):
        assert _status_for(report, "AB08", f"AB08_NOEXO_SLJ_L_{trial_number}") == discovery.OK
