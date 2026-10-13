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
    # Coordinate.setValue() defaults to enforceConstraints=True, which runs a
    # full assembly solve PER COORDINATE. With ~33 coordinates that is ~33
    # redundant solves per frame; measured at 265 ms/frame, i.e. ~10 minutes
    # for a single 3000-frame trial. Setting all coordinates first and
    # assembling ONCE per frame is mathematically the same (verified: max COM
    # difference 4.0e-08 m over a real trial) and ~32x faster.
    #
    # Skipping assembly altogether is ~45000x faster but NOT safe here: it
    # leaves the patellofemoral coupler constraints unsatisfied, giving a
    # 2.9e-05 m per-frame COM error. That is invisible in position but this
    # COM gets double-differentiated for GRF, where a per-frame error of that
    # size becomes ~0.9 m/s^2 -- around 63 N for a 70 kg subject.
    handles = [coordinates.get(name) for name in names]
    result = np.zeros((len(ik), 3))
    for row in range(len(ik)):
        for handle, value in zip(handles, values[row]):
            handle.setValue(state, float(value), False)
        model.assemble(state)
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
    # Same per-coordinate assembly problem as center_of_mass() above -- see
    # the comment there. Positions are set first and assembled once, then
    # speeds are applied, since assembly adjusts coordinate VALUES only.
    handles = [coordinates.get(name) for name in names]
    for row in range(len(time)):
        state.setTime(float(time[row]))
        for handle, value in zip(handles, values[row]):
            handle.setValue(state, float(value), False)
        model.assemble(state)
        for handle, speed in zip(handles, speeds[row]):
            handle.setSpeedValue(state, float(speed))
        model.realizeVelocity(state)
        position = model.calcMassCenterPosition(state)
        angular = model.calcAngularMomentum(state)
        com[row] = [position.get(i) for i in range(3)]
        momentum[row] = [angular.get(i) for i in range(3)]
    momentum = lowpass(momentum, time, cutoff_hz)
    return com, np.gradient(momentum, time, axis=0)
