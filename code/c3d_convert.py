"""C3dConverter: raw Vicon C3D -> marker TRC + raw six-plate force MOT.

Force location 1 is OpenSim's moving center of pressure. It must be selected
before reading the C3D; the default location is the fixed plate origin,
which is why ForcePlateProcessor.validate_moving_cop exists as a guard.
"""
from pathlib import Path
import re

import numpy as np

from config import DEFAULT
from opensim_env import opensim_module


class C3dConverter:
    def __init__(self, config=DEFAULT):
        self.config = config

    @staticmethod
    def _skip(name: str) -> bool:
        lowered = name.lower()
        return "discard" in lowered or "_old" in lowered or bool(re.search(r"_\d{3}$", name))

    def _rotate_table(self, table):
        angle = np.deg2rad(self.config.c3d_rotation_degrees)
        cos, sin = np.cos(angle), np.sin(angle)
        rotations = {
            "x": np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]]),
            "y": np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]),
            "z": np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]),
        }
        rotation = rotations[self.config.c3d_rotation_axis.lower()]
        matrix = table.updMatrix()
        for row in range(matrix.nrow()):
            for column in range(matrix.ncol()):
                value = matrix.getElt(row, column)
                vector = rotation @ np.array([value.get(index) for index in range(3)])
                target = matrix.updElt(row, column)
                for index in range(3):
                    target.set(index, float(vector[index]))

    def convert(self, subject, condition: str) -> list[Path]:
        """Convert every non-discarded C3D for one subject/condition."""
        osim = opensim_module()
        raw = subject.data_folder / "RAW" / "MOCAP"
        processed = subject.data_folder / "PROCESSED" / "MOCAP"
        files = (
            list(raw.glob(f"{subject.id}_{condition.upper()}_*.c3d"))
            + list(raw.glob(f"{subject.id}_NOEXO_static*.c3d"))
        )
        converted = []
        for c3d in dict.fromkeys(files):
            name = c3d.stem
            if self._skip(name):
                continue

            trc = processed / "marker_data" / f"{name}_markers_PROCESSED.trc"
            all_forces = processed / "force_data" / f"{name}_forces_ALL.mot"
            already_done = trc.is_file() and ("static" in name.lower() or all_forces.is_file())
            if already_done and not self.config.overwrite:
                converted.append(trc)
                continue

            trc.parent.mkdir(parents=True, exist_ok=True)
            all_forces.parent.mkdir(parents=True, exist_ok=True)

            adapter = osim.C3DFileAdapter()
            adapter.setLocationForForceExpression(1)
            tables = adapter.read(str(c3d))
            markers = adapter.getMarkersTable(tables)
            self._rotate_table(markers)
            osim.TRCFileAdapter.write(markers, str(trc))

            text = trc.read_text(errors="replace").replace("_x0005F_", "_").replace("Skeleton_", "")
            # Some Vicon sessions prefix every marker with an acquisition ID such as 002_.
            text = re.sub(r"(?m)(?<=\t)\d+_(?=[A-Za-z])", "", text)
            trc.write_text(text)

            if "static" not in name.lower():
                forces = adapter.getForcesTable(tables)
                self._rotate_table(forces)
                flat = forces.flatten()
                osim.STOFileAdapter.write(flat, str(all_forces))

            converted.append(trc)
        return converted
