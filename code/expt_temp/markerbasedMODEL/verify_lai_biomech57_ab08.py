"""verify_lai_biomech57_ab08.py -- compare the corrected LaiUhlrich2022 +
Biomech57 scaled model (build_lai_biomech57_model.py +
scale_lai_biomech57_ab08.py) against:
  (a) the CURRENT (wrong) Rajagopal-based scaled model this pipeline has
      actually been using (results/AB08/marker_based/models/
      AB08_shoe_size_5_NOEXO_scaled.osim)
  (b) OpenCap's own AB08 NOEXO scaled model (LaiUhlrich2022_scaled.osim,
      the real thing the framework pipeline runs IK/ID against)

Compares: total body mass, per-segment mass for major segments, and
marker-tracking RMS error at the static pose (does the corrected model's
Biomech57 markerset actually land close to the real marker positions in
AB08's own NOEXO static trial, i.e. did scaling converge sensibly).

Run from code/, in the opensimPy environment:
    python expt_temp/markerbasedMODEL/verify_lai_biomech57_ab08.py
"""
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from config import DEFAULT  # noqa: E402
from io_tables import read_storage, read_trc  # noqa: E402
from opensim_env import opensim_module  # noqa: E402
from subject import Subject  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
NEW_MODEL = OUT_DIR / "AB08_NOEXO_LaiUhlrich2022_Biomech57_scaled.osim"
NEW_STATIC_POSE = OUT_DIR / "StaticPose.mot"
CURRENT_WRONG_MODEL = CODE_ROOT.parent / "results" / "AB08" / "marker_based" / "models" / "AB08_shoe_size_5_NOEXO_scaled.osim"
OPENCAP_MODEL = (
    CODE_ROOT.parent / "data" / "AB08" / "PROCESSED" / "OPENCAP"
    / "OpenCapData_b3e1ba59-604c-4d22-af2d-579ced1c0c58" / "OpenSimData" / "Model" / "LaiUhlrich2022_scaled.osim"
)


def _total_and_segment_mass(osim, model_path):
    model = osim.Model(str(model_path))
    model.finalizeConnections()
    bodies = model.getBodySet()
    masses = {}
    total = 0.0
    for index in range(bodies.getSize()):
        body = bodies.get(index)
        masses[body.getName()] = body.getMass()
        total += body.getMass()
    return total, masses


def _marker_rms(osim, model_path, trc_path, static_pose_path):
    """RMS distance (mm) between each model marker's location AT THE
    MARKERPLACER-FITTED STATIC POSE (StaticPose.mot, the pose ScaleTool
    itself solved for and used to compute scale factors -- NOT the model's
    default/reference coordinates, which are unrelated to the subject's
    actual lab-frame position/orientation and give meaningless multi-meter
    "errors") and its experimental counterpart's mean position over the
    static trial."""
    model = osim.Model(str(model_path))
    state = model.initSystem()

    # StaticPose.mot's header says inDegrees=no -- values are already the
    # coordinate's native units (radians for rotational, meters for
    # translational), so no unit conversion is needed here.
    pose = read_storage(static_pose_path)
    coordinate_set = model.getCoordinateSet()
    for column in pose.columns:
        if column == "time" or not column.endswith("/value"):
            continue
        name = column.split("/")[-2]
        if not coordinate_set.contains(name):
            continue
        coordinate_set.get(name).setValue(state, float(pose[column].iloc[0]))
    model.realizePosition(state)

    trc = read_trc(trc_path)
    marker_names = trc["marker_names"]
    xyz = trc["xyz"]  # frames x markers x 3, meters

    marker_set = model.getMarkerSet()
    model_marker_names = {marker_set.get(i).getName() for i in range(marker_set.getSize())}

    distances = []
    for name in model_marker_names:
        if name not in marker_names:
            continue
        marker_index = marker_names.index(name)
        experimental = np.nanmean(xyz[:, marker_index, :], axis=0)
        if np.any(np.isnan(experimental)):
            continue
        model_marker = marker_set.get(name)
        model_point = model_marker.getLocationInGround(state).to_numpy()
        distances.append(np.linalg.norm(model_point - experimental))
    return float(np.sqrt(np.mean(np.square(distances)))) * 1000, len(distances), len(model_marker_names)


def main():
    osim = opensim_module()
    subject = Subject("AB08", DEFAULT)
    noexo_static = (
        subject.data_folder / "PROCESSED" / "MOCAP" / "marker_data"
        / "AB08_NOEXO_static_markers_PROCESSED.trc"
    )

    print("=" * 70)
    print("MODEL IDENTITY / MASS COMPARISON")
    print("=" * 70)
    rows = []
    for label, path in [
        ("NEW (LaiUhlrich2022+Biomech57)", NEW_MODEL),
        ("CURRENT (Rajagopal, wrong)", CURRENT_WRONG_MODEL),
        ("OpenCap (LaiUhlrich2022_scaled)", OPENCAP_MODEL),
    ]:
        if not Path(path).is_file():
            print(f"{label}: MISSING -> {path}")
            continue
        total, masses = _total_and_segment_mass(osim, path)
        rows.append((label, total, masses))
        print(f"\n{label} ({path.name}):")
        print(f"  total mass: {total:.3f} kg  (AB08 measured mass: {subject.mass_kg} kg)")
        for segment in ("pelvis", "femur_r", "tibia_r", "torso", "calcn_r"):
            if segment in masses:
                print(f"  {segment}: {masses[segment]:.3f} kg")

    print()
    print("=" * 70)
    print("MARKER-TRACKING RMS AT STATIC POSE (NEW model only -- OpenCap/")
    print("Rajagopal models use a different/no Biomech57 markerset, not comparable)")
    print("=" * 70)
    if NEW_MODEL.is_file() and noexo_static.is_file() and NEW_STATIC_POSE.is_file():
        rms_mm, matched, total_markers = _marker_rms(osim, NEW_MODEL, noexo_static, NEW_STATIC_POSE)
        print(f"RMS = {rms_mm:.1f} mm over {matched}/{total_markers} markers matched to the static TRC")
    else:
        print("Skipped -- missing model or static TRC")


if __name__ == "__main__":
    main()
