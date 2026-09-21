"""The one place that imports opensim and sets its logging level."""
import numpy as np
from scipy.signal import butter, sosfiltfilt

from config import DEFAULT


def opensim_module(config=DEFAULT):
    try:
        import opensim
    except ImportError as error:
        raise RuntimeError(
            "OpenSim is unavailable. Activate the opensimPy venv, then run this again."
        ) from error
    opensim.Logger.setLevelString(config.opensim_log_level)
    return opensim


def center_of_mass(model_path, ik):
    """Per-frame whole-body center of mass and total model mass from an IK motion.

    `ik` is a DataFrame with a 'time' column plus one column per coordinate
    that appears in the model (extra columns, e.g. pose-only markers'
    coordinates, are ignored).
    """
    osim = opensim_module()
    model = osim.Model(str(model_path))
    state = model.initSystem()
    coordinates = model.getCoordinateSet()
    names = [name for name in ik.columns if name != "time" and coordinates.contains(name)]
    values = ik[names].to_numpy(float).copy()
    rotations = [i for i, name in enumerate(names) if name not in ("pelvis_tx", "pelvis_ty", "pelvis_tz")]
    if rotations and np.nanmax(np.abs(values[:, rotations])) > 6.5:
        values[:, rotations] = np.deg2rad(values[:, rotations])
    mass = sum(model.getBodySet().get(i).getMass() for i in range(model.getBodySet().getSize()))
    result = np.zeros((len(ik), 3))
    for row in range(len(ik)):
        for column, name in enumerate(names):
            coordinates.get(name).setValue(state, float(values[row, column]))
        model.realizePosition(state)
        position = model.calcMassCenterPosition(state)
        result[row] = [position.get(i) for i in range(3)]
    return result, mass


def lowpass(values, time, cutoff_hz):
    """Fourth-order zero-phase low-pass filter."""
    sample_rate = 1.0 / float(np.median(np.diff(time)))
    sections = butter(4, cutoff_hz, fs=sample_rate, output="sos")
    return sosfiltfilt(sections, values, axis=0)


def center_of_mass_and_momentum_rate(model_path, ik, time, cutoff_hz=4.5):
    """Model COM and whole-body angular-momentum rate (H-dot) from filtered IK.

    H-dot about the COM equals the net external moment about the COM. Paired
    with a known COP and vertical force, this lets the horizontal GRF
    components be solved from a moment-balance equation instead of relying
    solely on noisy horizontal COM double-differentiation -- see
    framework_processor.py's rotation-force correction.
    """
    osim = opensim_module()
    model = osim.Model(str(model_path))
    state = model.initSystem()
    coordinates = model.getCoordinateSet()
    names = [coordinates.get(i).getName() for i in range(coordinates.getSize())]
    source_time = ik.time.to_numpy(float)
    values = np.zeros((len(time), len(names)))
    for column, name in enumerate(names):
        if name in ik:
            values[:, column] = np.interp(time, source_time, ik[name])
        if name not in ("pelvis_tx", "pelvis_ty", "pelvis_tz"):
            values[:, column] = np.deg2rad(values[:, column])
    values = lowpass(values, time, cutoff_hz)
    speeds = np.gradient(values, time, axis=0)
    com = np.zeros((len(time), 3))
    momentum = np.zeros((len(time), 3))
    for row in range(len(time)):
        state.setTime(float(time[row]))
        for column, name in enumerate(names):
            coordinate = coordinates.get(name)
            coordinate.setValue(state, float(values[row, column]))
            coordinate.setSpeedValue(state, float(speeds[row, column]))
        model.realizeVelocity(state)
        position = model.calcMassCenterPosition(state)
        angular = model.calcAngularMomentum(state)
        com[row] = [position.get(i) for i in range(3)]
        momentum[row] = [angular.get(i) for i in range(3)]
    momentum = lowpass(momentum, time, cutoff_hz)
    return com, np.gradient(momentum, time, axis=0)
