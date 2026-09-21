"""InsoleCalibration must fit gain_L and gain_R independently.

Regression test for the bug in resources/_old_reference_code: it fit ONE
gain from combined bilateral (left+right) stable-stance force and applied
it to both feet. This constructs a fully synthetic subject with two
different true per-side gains and asserts the calibration recovers both,
distinctly and correctly.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration import InsoleCalibration
from config import Config
from subject import Subject

MASS_KG = 70.0
GRAVITY = 9.80665
WEIGHT_N = MASS_KG * GRAVITY
TRUE_GAIN_L = 0.8
TRUE_GAIN_R = 1.3

INSOLE_HEADER = (
    "# time\tLeft Total Force\tRight Total Force\t"
    "Left Center of Pressure X\tLeft Center of Pressure Y\t"
    "Right Center of Pressure X\tRight Center of Pressure Y\n"
)


def _write_insole(path, time, left_force, right_force):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        file.write("synthetic insole export, for test_calibration.py only\n")
        file.write(INSOLE_HEADER)
        for t, left, right in zip(time, left_force, right_force):
            file.write(f"{t}\t{left}\t{right}\t0\t0\t0\t0\n")


def _default_subject_rules():
    return {
        "force_scale_mode": {"L": "none", "R": "none"},
        "force_scale": {"L": 1.0, "R": 1.0},
        "exclude": [], "use_last_flight_peak": [], "landing_ordinal": {},
        "opencap_aliases": {}, "insole_aliases": {}, "opencap_session_overrides": {},
        "repaired_trials": [], "insole_vertical_weight": 1.0, "com_vertical_weight": 0.0,
    }


@pytest.fixture
def synthetic_subject(tmp_path):
    subject_id = "TS99"
    data_root = tmp_path / "data"
    mocap = data_root / subject_id / "RAW" / "MOCAP"
    insole = data_root / subject_id / "RAW" / "INSOLE"
    mocap.mkdir(parents=True)
    insole.mkdir(parents=True)

    (mocap / f"{subject_id}_NOEXO_SLJ_L_1.c3d").write_bytes(b"")
    (mocap / f"{subject_id}_NOEXO_SLJ_R_1.c3d").write_bytes(b"")

    # A stable plateau (samples 100:300) at a known raw value, zero elsewhere
    # (zero is < 25% of body weight, so it's excluded from the stable window
    # by construction, exactly like a flight phase would be).
    n = 400
    time = np.arange(n) * 0.005
    left_raw = np.zeros(n)
    left_raw[100:300] = WEIGHT_N / TRUE_GAIN_L
    right_raw = np.zeros(n)
    right_raw[100:300] = WEIGHT_N / TRUE_GAIN_R
    _write_insole(insole / f"{subject_id}_NOEXO_SLJ_L_1.txt", time, left_raw, np.zeros(n))
    _write_insole(insole / f"{subject_id}_NOEXO_SLJ_R_1.txt", time, np.zeros(n), right_raw)

    (data_root / "subject_metadata.csv").write_text(
        "SubjectID,Mass_kg,Height_m,ExoLeg,ShoeSize\n"
        f"{subject_id},{MASS_KG},1.75,R,7\n"
    )

    subject_rules_root = tmp_path / "subject_rules"
    subject_rules_root.mkdir()
    (subject_rules_root / "default.json").write_text(json.dumps(_default_subject_rules()))

    config = Config(project_root=tmp_path, data_root=data_root)
    return Subject(subject_id, config), config


def test_gain_fit_independently_per_side(synthetic_subject):
    subject, config = synthetic_subject
    calibration = InsoleCalibration(config)
    auto_rules = {
        "force_scale_mode": {"L": InsoleCalibration.AUTO_MODE, "R": InsoleCalibration.AUTO_MODE},
        "force_scale": {"L": 1.0, "R": 1.0},
    }

    gain_l, source_l, weight_l, median_l, count_l, reports_l = calibration._force_gain(subject, "L", auto_rules)
    gain_r, source_r, weight_r, median_r, count_r, reports_r = calibration._force_gain(subject, "R", auto_rules)

    # rel=1e-2, not exact: the stance window is now taken from a 6Hz low-pass
    # filtered signal (see InsoleCalibration._stance_window), which smooths
    # the synthetic plateau's edges slightly rather than reproducing it exactly.
    assert gain_l == pytest.approx(TRUE_GAIN_L, rel=1e-2)
    assert gain_r == pytest.approx(TRUE_GAIN_R, rel=1e-2)
    assert gain_l != gain_r, "gain_L and gain_R must not collapse to a shared value"
    assert source_l == source_r == "body_weight_over_median_of_accepted_trial_stance_medians"
    assert count_l == count_r == 1  # one trial per side, its stance window accepted
    assert reports_l[0]["Used"] and reports_r[0]["Used"]


def test_fixed_mode_is_still_per_side(synthetic_subject):
    subject, _config = synthetic_subject
    calibration = InsoleCalibration(_config)
    rules = {
        "force_scale_mode": {"L": "fixed", "R": "fixed"},
        "force_scale": {"L": 0.5, "R": 0.9},
    }
    gain_l, source_l, *_ = calibration._force_gain(subject, "L", rules)
    gain_r, source_r, *_ = calibration._force_gain(subject, "R", rules)
    assert (gain_l, source_l) == (0.5, "fixed_value")
    assert (gain_r, source_r) == (0.9, "fixed_value")
