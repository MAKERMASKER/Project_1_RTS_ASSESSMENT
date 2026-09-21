"""SingleLegJump.synchronize() must pick the correct flight/landing/contact
from a known synthetic toe-height + insole-force signal, with no OpenSim
dependency (no real TRC/MOT files needed)."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from activities import SingleLegJump
from config import DEFAULT
from subject import Trial

BASELINE_M = 0.05
FLIGHT_APEX_S = 1.05
FLIGHT_HALF_WIDTH_S = 0.05  # apex at 1.05, base at 1.00 and 1.10
LANDING_START_S = 1.10
CONTACT_RAMP_S = 0.10       # 0 -> plateau over 0.10 s
PLATEAU_START_S = 1.25
PLATEAU_END_S = 1.45
PLATEAU_FORCE_N = 1200.0
DT = 0.01


def _synthetic_trc(side: str) -> dict:
    time = np.arange(0, 3.0 + DT, DT)
    height = np.full_like(time, BASELINE_M)
    in_flight = (time >= 1.00) & (time <= FLIGHT_APEX_S + FLIGHT_HALF_WIDTH_S)
    height[in_flight] = BASELINE_M + 0.15 * (1 - np.abs(time[in_flight] - FLIGHT_APEX_S) / FLIGHT_HALF_WIDTH_S)

    toe1_name = "L_toe_study" if side == "L" else "r_toe_study"
    toe5_name = "L_5meta_study" if side == "L" else "r_5meta_study"
    marker_names = [toe1_name, toe5_name]
    xyz = np.zeros((len(time), 2, 3))
    xyz[:, 0, 1] = height  # Y = vertical, same height on both toe markers
    xyz[:, 1, 1] = height
    return {"time": time, "marker_names": marker_names, "xyz": xyz}


def _synthetic_insole(side: str) -> pd.DataFrame:
    time = np.arange(0, 3.0 + DT, DT)
    force = np.zeros_like(time)
    rising = (time >= LANDING_START_S) & (time < PLATEAU_START_S)
    force[rising] = PLATEAU_FORCE_N * (time[rising] - LANDING_START_S) / (PLATEAU_START_S - LANDING_START_S)
    plateau = (time >= PLATEAU_START_S) & (time <= PLATEAU_END_S)
    force[plateau] = PLATEAU_FORCE_N
    falling = (time > PLATEAU_END_S) & (time <= PLATEAU_END_S + CONTACT_RAMP_S)
    force[falling] = PLATEAU_FORCE_N * (1 - (time[falling] - PLATEAU_END_S) / CONTACT_RAMP_S)

    column = "left_force" if side == "L" else "right_force"
    other = "right_force" if side == "L" else "left_force"
    return pd.DataFrame({
        "time": time, column: force, other: np.zeros_like(time),
        "left_cop_x": np.zeros_like(time), "left_cop_y": np.zeros_like(time),
        "right_cop_x": np.zeros_like(time), "right_cop_y": np.zeros_like(time),
    })


def _synthetic_ik(time: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"time": time, "pelvis_tx": np.zeros_like(time)})


def _trial(side: str) -> Trial:
    dummy = Path("unused")
    return Trial(
        subject="TS99", condition="NOEXO", activity="single_leg_jump",
        name=f"TS99_NOEXO_SLJ_{side}_1", side=side,
        insole=dummy, c3d=dummy, marker_trc=dummy, force_mot=dummy,
        opencap_trc=dummy, opencap_ik=dummy, opencap_model=dummy, output=dummy,
    )


@pytest.mark.parametrize("side", ["L", "R"])
def test_synchronize_picks_the_constructed_landing_and_peak(side):
    trc = _synthetic_trc(side)
    insole_all = _synthetic_insole(side)
    ik_all = _synthetic_ik(insole_all["time"].to_numpy())
    trial = _trial(side)
    rules = {"use_last_flight_peak": set(), "landing_ordinal": {}}

    activity = SingleLegJump(DEFAULT)
    time, insole, ik, sync = activity.synchronize(trial, trc, ik_all, insole_all, rules)

    assert sync["method"] == "complete_jump_around_landing_peak"
    assert sync["quality_omit"] is False
    # The force plateau spans [1.25, 1.45]; after 7-sample smoothing the peak
    # must still land inside (or immediately adjacent to) that plateau.
    assert PLATEAU_START_S - 0.03 <= sync["peak_grf_time_s"] <= PLATEAU_END_S + 0.03
    # The returned window is the configured +/-0.75 s around that peak.
    assert time.min() == pytest.approx(sync["peak_grf_time_s"] + DEFAULT.jump_window_s[0], abs=DT)
    assert time.max() == pytest.approx(sync["peak_grf_time_s"] + DEFAULT.jump_window_s[1], abs=DT)


def test_landing_ordinal_override_picks_a_different_contact():
    """A second, smaller, earlier contact should be ignored by default (only
    the peak-force contact is chosen), but MUST be picked when
    landing_ordinal explicitly selects it as the first chronological contact."""
    side = "L"
    trc = _synthetic_trc(side)
    insole_all = _synthetic_insole(side)
    time = insole_all["time"].to_numpy()

    # An isolated, smaller contact well before the flight/landing (0.30-0.60 s),
    # clearly separated in time from the main 1200 N plateau contact.
    rise = (time >= 0.30) & (time < 0.40)
    plateau = (time >= 0.40) & (time <= 0.50)
    fall = (time > 0.50) & (time <= 0.60)
    insole_all.loc[rise, "left_force"] = 300.0 * (time[rise] - 0.30) / 0.10
    insole_all.loc[plateau, "left_force"] = 300.0
    insole_all.loc[fall, "left_force"] = 300.0 * (1 - (time[fall] - 0.50) / 0.10)

    ik_all = _synthetic_ik(time)
    trial = _trial(side)

    default_rules = {"use_last_flight_peak": set(), "landing_ordinal": {}}
    activity = SingleLegJump(DEFAULT)
    _, _, _, default_sync = activity.synchronize(trial, trc, ik_all, insole_all, default_rules)
    assert PLATEAU_START_S - 0.03 <= default_sync["peak_grf_time_s"] <= PLATEAU_END_S + 0.03

    override_rules = {"use_last_flight_peak": set(), "landing_ordinal": {trial.name: 1}}
    _, _, _, override_sync = activity.synchronize(trial, trc, ik_all, insole_all, override_rules)
    # Both syncs re-anchor to the same marker-derived landing time, so the
    # *reported* peak time alone doesn't distinguish which contact was
    # chosen -- but the alignment shift does: the early (~0.43 s) and main
    # (~1.28 s) contacts differ in peak-to-contact-start offset by roughly
    # the ~0.77 s gap between their contact-start times.
    assert override_sync["shift_s"] == pytest.approx(default_sync["shift_s"] + 0.77, abs=0.03)
