"""InsoleGeometry's foot anatomical frame must be a true orthonormal
round-trip: projecting a lab-frame point into the frame and back must
recover the original point exactly."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insole_geometry import InsoleGeometry


def _static_foot_trc(side: str, n_frames: int = 5) -> dict:
    """A simple, static (non-moving) foot: heel behind, two forefoot markers ahead."""
    if side == "R":
        heel, toe1, toe5 = (1.00, 0.05, 0.00), (1.25, 0.03, 0.05), (1.25, 0.03, -0.05)
        names = ["r_calc_study", "r_toe_study", "r_5meta_study"]
    else:
        heel, toe1, toe5 = (1.00, 0.05, 0.00), (1.25, 0.03, -0.05), (1.25, 0.03, 0.05)
        names = ["L_calc_study", "L_toe_study", "L_5meta_study"]

    time = np.linspace(0, 1, n_frames)
    xyz = np.zeros((n_frames, 3, 3))
    for index, point in enumerate((heel, toe1, toe5)):
        xyz[:, index, :] = point
    return {"time": time, "marker_names": names, "xyz": xyz}


@pytest.mark.parametrize("side", ["L", "R"])
def test_to_local_round_trips_through_frames(side):
    trc = _static_foot_trc(side)
    times = trc["time"]
    points = np.array([
        [1.00, 0.90, 0.20],
        [1.05, 0.92, 0.18],
        [0.98, 0.88, 0.22],
        [1.10, 0.95, -0.05],
        [1.02, 0.91, 0.00],
    ])
    assert len(points) == len(times)

    local = InsoleGeometry.to_local(trc, times, points, side)
    origin, rotation = InsoleGeometry.frames(trc, times, side)
    reconstructed = origin + np.einsum("nij,nj->ni", rotation, local)

    np.testing.assert_allclose(reconstructed, points, atol=1e-9)


@pytest.mark.parametrize("side", ["L", "R"])
def test_frame_axes_are_orthonormal(side):
    trc = _static_foot_trc(side)
    _origin, rotation = InsoleGeometry.frames(trc, trc["time"], side)
    for frame in rotation:
        # Each column (ap, ml, vertical) is unit length and mutually orthogonal.
        np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-9)


def test_native_cop_and_transform_round_trip_with_zero_offset():
    """transform() maps native (insole-fraction) COP into the lab frame via
    the anatomical frame; with a zero rigid offset, re-deriving the native
    fractional coordinates from the transformed lab point must recover the
    original ap/ml fractions."""
    side = "R"
    trc = _static_foot_trc(side, n_frames=3)
    geometry = InsoleGeometry(shoe_size=7)
    ap = np.array([0.1, -0.2, 0.05])   # native AP fraction (insole-length units, offset by 0.5 internally)
    ml = np.array([0.05, -0.1, 0.0])   # native ML fraction
    insole = {"right_cop_x": ap, "right_cop_y": ml}
    placement = {"rigid_offset_local_m": [0.0, 0.0, 0.0]}

    lab_cop = geometry.transform(trc, trc["time"], insole, side, placement)
    local_cop = InsoleGeometry.to_local(trc, trc["time"], lab_cop, side)
    expected_native = geometry.native_cop(ap, ml, side)
    np.testing.assert_allclose(local_cop, expected_native, atol=1e-9)
