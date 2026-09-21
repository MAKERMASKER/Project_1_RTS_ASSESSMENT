"""Read/write for OpenSim MOT/STO/TRC tables, and the raw insole .txt format.

Binary/OpenSim-native formats are always delegated to OpenSim's own file
adapters rather than a hand-written parser. Plain tab-separated text (insole
exports) is read with pandas.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from opensim_env import opensim_module


def read_storage(path) -> pd.DataFrame:
    """Read an OpenSim MOT/STO file into a DataFrame with a 'time' column."""
    osim = opensim_module()
    table = osim.TimeSeriesTable(str(Path(path)))
    labels = list(table.getColumnLabels())
    time = np.asarray(list(table.getIndependentColumn()), dtype=float)
    values = np.asarray(table.getMatrix().to_numpy(), dtype=float).copy()
    return pd.DataFrame(np.column_stack([time, values]), columns=["time", *labels])


def write_storage(path, table: pd.DataFrame, title="OpenSim storage", in_degrees="no"):
    """Write a DataFrame with a 'time' column to an OpenSim MOT/STO file."""
    osim = opensim_module()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = osim.TimeSeriesTable()
    labels = [str(name) for name in table.columns if name != "time"]
    output.setColumnLabels(osim.StdVectorString(labels))
    output.addTableMetaDataString("name", title)
    output.addTableMetaDataString("inDegrees", in_degrees)
    values = table[labels].to_numpy(float)
    times = pd.to_numeric(table["time"], errors="raise").to_numpy(float)
    for time, values_at_time in zip(times, values):
        row = osim.RowVector(len(labels))
        for column, value in enumerate(values_at_time):
            row[column] = float(value)
        output.appendRow(float(time), row)
    osim.STOFileAdapter.write(output, str(path))


def write_bilateral_grf(path, time, right_force, right_cop, left_force, left_cop):
    """Write right as ground_force_1 (-> calcn_r) and left as force_2 (-> calcn_l)."""
    names = ["time"]
    for number in (1, 2):
        names += [f"ground_force_{number}_v{axis}" for axis in "xyz"]
        names += [f"ground_force_{number}_p{axis}" for axis in "xyz"]
        names += [f"ground_moment_{number}_m{axis}" for axis in "xyz"]
    zeros = np.zeros_like(right_force)
    values = np.column_stack([time, right_force, right_cop, zeros, left_force, left_cop, zeros])
    write_storage(path, pd.DataFrame(values, columns=names), "bilateral_opencap_insole_grf")


def read_trc(path) -> dict:
    """Read a TRC via OpenSim. Returns {'time', 'marker_names', 'xyz'}."""
    osim = opensim_module()
    table = osim.TimeSeriesTableVec3(str(Path(path)))
    labels = list(table.getColumnLabels())
    time = np.array(list(table.getIndependentColumn()), dtype=float)
    matrix = table.getMatrix()
    xyz = np.empty((len(time), len(labels), 3))
    for row in range(len(time)):
        for column in range(len(labels)):
            value = matrix.getElt(row, column)
            xyz[row, column] = [value.get(0), value.get(1), value.get(2)]
    return {"time": time, "marker_names": labels, "xyz": xyz}


def write_trc(source_path, destination_path, xyz):
    """Write a TRC by copying an existing TRC's structure and replacing XYZ values.

    Used by the TRC-repair subsystem: keeps every header/metadata field
    (marker names, units, frame rate) from the source file untouched.
    """
    osim = opensim_module()
    table = osim.TimeSeriesTableVec3(str(Path(source_path)))
    matrix = table.updMatrix()
    for row in range(matrix.nrow()):
        for column in range(matrix.ncol()):
            target = matrix.updElt(row, column)
            for axis in range(3):
                target.set(axis, float(xyz[row, column, axis]))
    osim.TRCFileAdapter.write(table, str(Path(destination_path)))


def interpolate_table(table: pd.DataFrame, times) -> pd.DataFrame:
    out = pd.DataFrame({"time": times})
    for name in table.columns:
        if name != "time":
            out[name] = np.interp(times, table.time, table[name], left=np.nan, right=np.nan)
    return out


def segments(mask) -> np.ndarray:
    """Contiguous True runs in a boolean mask as an (N, 2) [start, end] array."""
    changes = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return np.c_[np.flatnonzero(changes == 1), np.flatnonzero(changes == -1) - 1]


def read_insole(path) -> pd.DataFrame:
    """Read a raw insole .txt export.

    Format: a free-text header block, then one line starting with '# time'
    giving tab-separated column names, then tab-separated data. Force and
    COP columns are located by substring match rather than an exact name,
    since header wording has varied across export sessions.
    """
    path = Path(path)
    with path.open(errors="replace") as file:
        for count, line in enumerate(file, 1):
            if line.strip().startswith("# time"):
                names = line.strip()[1:].split("\t")
                break
        else:
            raise ValueError(f"Missing insole header line ('# time ...'): {path}")
    raw = pd.read_csv(path, sep="\t", skiprows=count, names=names)

    def find(*words):
        return next(
            column for column in raw
            if all(word in column.lower().replace(" ", "") for word in words)
        )

    left_cop = [c for c in raw if "leftcenterofpressure" in c.lower().replace(" ", "")]
    right_cop = [c for c in raw if "rightcenterofpressure" in c.lower().replace(" ", "")]
    out = pd.DataFrame({
        "time": raw.iloc[:, 0],
        "left_force": raw[find("lefttotalforce")],
        "right_force": raw[find("righttotalforce")],
        "left_cop_x": raw[left_cop[0]],
        "left_cop_y": raw[left_cop[1]],
        "right_cop_x": raw[right_cop[0]],
        "right_cop_y": raw[right_cop[1]],
    })
    return out.interpolate(limit_direction="both")
