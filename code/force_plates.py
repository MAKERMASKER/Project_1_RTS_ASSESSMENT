"""ForcePlateProcessor: right/left resultant marker-based GRF from raw force plates.

Written once; never changes when a new activity is added. Every activity
decision comes from activity.marker_plates(trial) -- a fixed plate list per
foot for most activities, or None to trigger the distance-based nearest-foot
assignment (needed by activities with no fixed plate, e.g. agility -- not
used by SingleLegJump today, kept because the Activity contract allows it).
"""
import shutil

import numpy as np
import pandas as pd

from config import DEFAULT
from io_tables import read_storage, read_trc, segments, write_storage


class ForcePlateProcessor:
    FORCE_THRESHOLD_N = 20.0
    MIN_CONTACT_DURATION_S = 0.020
    RIGHT_FOOT_MARKERS = ("RFCC", "RFM1", "RFM5")
    LEFT_FOOT_MARKERS = ("LFCC", "LFM1", "LFM5")

    def __init__(self, config=DEFAULT):
        self.config = config

    def process(self, activity, trial):
        """Build the two-plate (right=1, left=2) resultant force MOT for one trial."""
        target = trial.force_mot
        source = target.with_name(target.name.replace("_forces_PROCESSED.mot", "_forces_ALL.mot"))
        if not source.is_file():
            if not target.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(target, source)

        table = read_storage(source)
        fixed_rules = activity.marker_plates(trial)

        plates_to_check = list(range(1, 7)) if fixed_rules is None else sorted(set(fixed_rules[0] + fixed_rules[1]))
        self._validate_moving_cop(table, plates_to_check)

        if fixed_rules is None:
            right_masks, left_masks = self._assign_agility_contacts(table, trial)
            all_plates = list(range(1, 7))
            right_force, right_cop, right_moment = self._resultant(table, all_plates, right_masks)
            left_force, left_cop, left_moment = self._resultant(table, all_plates, left_masks)
        else:
            right_plates, left_plates = fixed_rules
            right_masks = self._contact_masks_for(activity, trial, table, right_plates)
            left_masks = self._contact_masks_for(activity, trial, table, left_plates)
            right_force, right_cop, right_moment = self._resultant(table, right_plates, right_masks)
            left_force, left_cop, left_moment = self._resultant(table, left_plates, left_masks)

        names = (
            ["time"]
            + [f"ground_force_1_v{axis}" for axis in "xyz"]
            + [f"ground_force_1_p{axis}" for axis in "xyz"]
            + [f"ground_moment_1_m{axis}" for axis in "xyz"]
            + [f"ground_force_2_v{axis}" for axis in "xyz"]
            + [f"ground_force_2_p{axis}" for axis in "xyz"]
            + [f"ground_moment_2_m{axis}" for axis in "xyz"]
        )
        values = np.column_stack([
            pd.to_numeric(table["time"], errors="coerce"),
            right_force, right_cop, right_moment,
            left_force, left_cop, left_moment,
        ])
        write_storage(target, pd.DataFrame(values, columns=names), "marker_based_resultant_forces")
        return target

    # ---- plate math ------------------------------------------------------

    def _contact_masks_for(self, activity, trial, table, plates):
        """Give the activity a chance to filter samples beyond this class's
        own force threshold (see Activity.contact_masks() -- a no-op
        returning None for every activity except LateralHop). Extracts
        each requested plate's own vertical force so the activity hook
        works with plain arrays, not this module's column-naming scheme."""
        if not plates:
            return None
        dt = float(np.median(np.diff(pd.to_numeric(table["time"], errors="coerce"))))
        plate_forces = {
            plate: self._numbers(table, self._plate_columns(table, plate, "force"))[:, 1]
            for plate in plates
        }
        return activity.contact_masks(trial, plate_forces, dt)

    @staticmethod
    def _numbers(table, names):
        return table[names].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float, copy=True)

    @staticmethod
    def _plate_columns(table, plate, quantity):
        axes = "xyz"
        long_names = {
            "force": [f"ground_force_{plate}_v{axis}" for axis in axes],
            "point": [f"ground_force_{plate}_p{axis}" for axis in axes],
            "moment": [f"ground_moment_{plate}_m{axis}" for axis in axes],
        }[quantity]
        prefix = {"force": "f", "point": "p", "moment": "m"}[quantity]
        short_names = [f"{prefix}{plate}_{number}" for number in (1, 2, 3)]
        for names in (long_names, short_names):
            if all(name in table.columns for name in names):
                return names
        raise KeyError(
            f"Force plate {plate} has no {quantity} columns. Expected {long_names} or {short_names}."
        )

    def _resultant(self, table, plates, masks=None):
        force_sum = np.zeros((len(table), 3))
        weighted_cop = np.zeros_like(force_sum)
        moment_at_origin = np.zeros_like(force_sum)
        vertical_weight = np.zeros(len(table))

        for plate in plates:
            force = self._numbers(table, self._plate_columns(table, plate, "force"))
            cop = self._numbers(table, self._plate_columns(table, plate, "point"))
            moment = self._numbers(table, self._plate_columns(table, plate, "moment"))
            contact = np.abs(force[:, 1]) >= self.FORCE_THRESHOLD_N
            if masks is not None:
                contact &= masks[plate]
            force[~contact] = 0
            cop[~contact] = 0
            moment[~contact] = 0
            force_sum += force
            moment_at_origin += moment + np.cross(cop, force)
            weight = np.abs(force[:, 1])
            vertical_weight += weight
            weighted_cop += cop * weight[:, None]

        combined_cop = np.zeros_like(force_sum)
        in_contact = vertical_weight > 0
        combined_cop[in_contact] = weighted_cop[in_contact] / vertical_weight[in_contact, None]
        combined_moment = moment_at_origin - np.cross(combined_cop, force_sum)
        force_sum[~in_contact] = 0
        combined_moment[~in_contact] = 0
        return force_sum, combined_cop, combined_moment

    def _validate_moving_cop(self, table, plates):
        """Guard against a C3D converted without a moving center-of-pressure."""
        checked = 0
        moving = 0
        for plate in plates:
            force = self._numbers(table, self._plate_columns(table, plate, "force"))
            cop = self._numbers(table, self._plate_columns(table, plate, "point"))
            contact = np.abs(force[:, 1]) >= self.FORCE_THRESHOLD_N
            if np.count_nonzero(contact) < 2:
                continue
            checked += 1
            floor_range = np.ptp(cop[contact][:, [0, 2]], axis=0)
            if np.nanmax(floor_range) > 1e-5:
                moving += 1
        if checked and not moving:
            raise ValueError(
                "Force-plate COP is fixed at the plate origin. Reconvert the C3D "
                "with C3DFileAdapter.setLocationForForceExpression(1)."
            )

    def _foot_markers_at_force_times(self, trial, force_times, marker_names):
        trc = read_trc(trial.marker_trc)
        positions = []
        for marker_name in marker_names:
            if marker_name not in trc["marker_names"]:
                raise KeyError(f"Marker {marker_name} is missing from {trial.marker_trc}")
            marker_index = trc["marker_names"].index(marker_name)
            marker_xyz = trc["xyz"][:, marker_index, :]
            interpolated = np.column_stack([
                np.interp(force_times, trc["time"], marker_xyz[:, axis]) for axis in range(3)
            ])
            positions.append(interpolated)
        return np.stack(positions, axis=1)

    def _assign_agility_contacts(self, table, trial):
        """Assign every plate contact with no fixed rule to the closer foot."""
        times = pd.to_numeric(table["time"], errors="coerce").to_numpy(float)
        right_foot = self._foot_markers_at_force_times(trial, times, self.RIGHT_FOOT_MARKERS)
        left_foot = self._foot_markers_at_force_times(trial, times, self.LEFT_FOOT_MARKERS)
        right_masks, left_masks = {}, {}

        for plate in range(1, 7):
            force = self._numbers(table, self._plate_columns(table, plate, "force"))
            cop = self._numbers(table, self._plate_columns(table, plate, "point"))
            active = np.abs(force[:, 1]) >= self.FORCE_THRESHOLD_N
            right_masks[plate] = np.zeros(len(table), dtype=bool)
            left_masks[plate] = np.zeros(len(table), dtype=bool)

            for start, end in segments(active):
                if times[end] - times[start] < self.MIN_CONTACT_DURATION_S:
                    continue
                event = slice(start, end + 1)
                cop_floor = cop[event][:, [0, 2]]
                right_floor = right_foot[event][:, :, [0, 2]]
                left_floor = left_foot[event][:, :, [0, 2]]
                right_distance = np.nanmedian(
                    np.nanmin(np.linalg.norm(right_floor - cop_floor[:, None, :], axis=2), axis=1)
                )
                left_distance = np.nanmedian(
                    np.nanmin(np.linalg.norm(left_floor - cop_floor[:, None, :], axis=2), axis=1)
                )
                if not np.isfinite(right_distance) or not np.isfinite(left_distance):
                    raise ValueError(f"Cannot pair plate {plate} contact with foot markers")
                leg_masks = right_masks if right_distance < left_distance else left_masks
                leg_masks[plate][event] = True
        return right_masks, left_masks
