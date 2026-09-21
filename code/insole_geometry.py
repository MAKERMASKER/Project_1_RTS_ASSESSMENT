"""Insole physical geometry and the foot anatomical frame <-> insole COP transform.

Ported from resources/_old_reference_code/insole.py. The foot-marker lookup
and the anatomical-frame construction don't depend on insole dimensions, so
they're usable (as classmethods) by activities.py's synchronization logic,
which never needs a shoe size. Only the native-COP <-> lab-frame transform
needs the measured insole dimensions, so that part requires an instance.
"""
import numpy as np


class InsoleGeometry:
    """Foot marker lookup, anatomical frame, and insole COP <-> lab transforms."""

    # Marker name aliases: Vicon ground-truth naming vs. OpenCap/OpenSim naming
    # differ, so every alias is tried in order.
    MARKERS = {
        "R": {
            "heel": ("r_calc_study", "RHeel", "RCalc", "RFCC"),
            "toe1": ("r_toe_study", "RBigToe", "RFM1"),
            "toe5": ("r_5meta_study", "RSmallToe", "RFM5"),
        },
        "L": {
            "heel": ("L_calc_study", "LHeel", "LCalc", "LFCC"),
            "toe1": ("L_toe_study", "LBigToe", "LFM1"),
            "toe5": ("L_5meta_study", "LSmallToe", "LFM5"),
        },
    }

    # Measured insole dimensions by shoe size (meters).
    DIMENSIONS = {
        5: {"size": 5, "length_m": 0.2611, "width_m": 0.0938},
        7: {"size": 7, "length_m": 0.2880, "width_m": 0.1014},
    }

    def __init__(self, shoe_size: int):
        if shoe_size not in (5, 7):
            raise ValueError(f"shoe_size must be 5 or 7; found {shoe_size}")
        self.shoe_size = shoe_size
        self.shape = dict(self.DIMENSIONS[shoe_size])

    @classmethod
    def for_subject(cls, subject) -> "InsoleGeometry":
        return cls(subject.shoe_size)

    @classmethod
    def marker(cls, trc, times, side, role):
        lower = [name.lower() for name in trc["marker_names"]]
        name = next((n for n in cls.MARKERS[side][role] if n.lower() in lower), None)
        if name is None:
            raise ValueError(f"Missing {side} {role} marker")
        xyz = trc["xyz"][:, lower.index(name.lower()), :].copy()
        if np.nanmedian(np.abs(xyz)) > 2:
            xyz /= 1000  # mm -> m
        return np.column_stack([
            np.interp(times, trc["time"], xyz[:, axis], left=np.nan, right=np.nan)
            for axis in range(3)
        ])

    @classmethod
    def foot_center(cls, trc, side):
        return np.mean(
            [cls.marker(trc, trc["time"], side, role) for role in ("heel", "toe1", "toe5")],
            axis=0,
        )

    @classmethod
    def foot_height(cls, trc, side):
        return np.nanmean(
            np.c_[
                cls.marker(trc, trc["time"], side, "toe1")[:, 1],
                cls.marker(trc, trc["time"], side, "toe5")[:, 1],
            ],
            axis=1,
        )

    @classmethod
    def frames(cls, trc, times, side):
        """Per-frame foot anatomical rotation matrix + origin (heel marker)."""
        heel, toe1, toe5 = (cls.marker(trc, times, side, role) for role in ("heel", "toe1", "toe5"))
        ap = 0.5 * (toe1 + toe5) - heel
        ap /= np.linalg.norm(ap, axis=1)[:, None]
        ml = toe5 - toe1
        ml -= np.sum(ml * ap, axis=1)[:, None] * ap
        ml /= np.linalg.norm(ml, axis=1)[:, None]
        vertical = np.cross(ap, ml)
        vertical /= np.linalg.norm(vertical, axis=1)[:, None]
        flip = vertical[:, 1] < 0
        vertical[flip] *= -1
        ml[flip] *= -1
        rotation = np.stack([ap, ml, vertical], axis=2)

        # Prevent a simultaneous AP/ML orientation reversal between frames.
        for index in range(1, len(rotation)):
            current, previous = rotation[index], rotation[index - 1]
            if not np.isfinite(np.c_[current, previous]).all():
                continue
            if np.dot(current[:, 0], previous[:, 0]) < 0 and np.dot(current[:, 1], previous[:, 1]) < 0:
                current[:, :2] *= -1
                current[:, 2] = np.cross(current[:, 0], current[:, 1])
        return heel, rotation

    def native_cop(self, ap, ml, side):
        """Insole-native fractional COP -> a 3D point in the insole's own frame."""
        return np.c_[
            (np.asarray(ap) + 0.5) * self.shape["length_m"],
            (1 if side == "R" else -1) * np.asarray(ml) * self.shape["width_m"],
            np.zeros(len(np.asarray(ap))),
        ]

    def transform(self, trc, times, insole, side, placement):
        """Insole-native COP -> lab frame, via the foot anatomical frame."""
        prefix = "right" if side == "R" else "left"
        local = self.native_cop(insole[f"{prefix}_cop_x"], insole[f"{prefix}_cop_y"], side)
        local += np.asarray(placement["rigid_offset_local_m"])
        origin, rotation = self.frames(trc, times, side)
        return origin + np.einsum("nij,nj->ni", rotation, local)

    @classmethod
    def to_local(cls, trc, times, points, side):
        """Lab-frame points (e.g. force-plate COP) -> the foot anatomical frame."""
        origin, rotation = cls.frames(trc, times, side)
        return np.einsum("nji,nj->ni", rotation, np.asarray(points) - origin)
