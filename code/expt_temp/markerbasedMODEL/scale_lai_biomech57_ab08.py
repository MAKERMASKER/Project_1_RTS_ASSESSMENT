"""scale_lai_biomech57_ab08.py -- scale the corrected LaiUhlrich2022 +
Biomech57 model (see build_lai_biomech57_model.py) for AB08's NOEXO static
trial, reusing the production ScaleTool machinery
(ModelScaler._write_scale_setup / the real Scale_NOEXO.xml template) so
this is a genuine apples-to-apples scaling run, not a reimplementation.

Run from code/, in the opensimPy environment:
    python expt_temp/markerbasedMODEL/scale_lai_biomech57_ab08.py
"""
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from config import DEFAULT  # noqa: E402
from opensim_env import opensim_module  # noqa: E402
from scale_ik_id import ModelScaler  # noqa: E402
from subject import Subject  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
UNSCALED_MODEL = OUT_DIR / "LaiUhlrich2022_Biomech57_unscaled.osim"
SCALED_OUTPUT = OUT_DIR / "AB08_NOEXO_LaiUhlrich2022_Biomech57_scaled.osim"
SCALE_FACTORS_OUTPUT = OUT_DIR / "AB08_scale_factors.xml"
SETUP_OUTPUT = OUT_DIR / "Setup_Scale_NOEXO_LaiBiomech57.xml"


def main():
    if not UNSCALED_MODEL.is_file():
        raise FileNotFoundError(f"Run build_lai_biomech57_model.py first -- missing {UNSCALED_MODEL}")

    subject = Subject("AB08", DEFAULT)
    scaler = ModelScaler(DEFAULT)
    noexo_static = (
        subject.data_folder / "PROCESSED" / "MOCAP" / "marker_data"
        / f"{subject.id}_NOEXO_static_markers_PROCESSED.trc"
    )
    if not noexo_static.is_file():
        raise FileNotFoundError(noexo_static)

    scaler._write_scale_setup(
        DEFAULT.scale_templates_root / "Scale_NOEXO.xml",
        SETUP_OUTPUT, UNSCALED_MODEL, noexo_static,
        SCALED_OUTPUT, SCALE_FACTORS_OUTPUT, subject.mass_kg, subject.height_m,
    )
    print(f"Setup written: {SETUP_OUTPUT}")

    osim = opensim_module()
    ok = osim.ScaleTool(str(SETUP_OUTPUT)).run()
    if not ok:
        raise RuntimeError("ScaleTool.run() returned False -- scaling failed")
    print(f"Scaled model written: {SCALED_OUTPUT}")

    scaler._unlock_coordinates(SCALED_OUTPUT)

    model = osim.Model(str(SCALED_OUTPUT))
    model.finalizeConnections()
    print(f"Load OK: {model.getMarkerSet().getSize()} markers, {model.getBodySet().getSize()} bodies, "
          f"{model.getCoordinateSet().getSize()} coordinates")


if __name__ == "__main__":
    main()
