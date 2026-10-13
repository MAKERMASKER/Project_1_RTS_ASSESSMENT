"""build_lai_biomech57_model.py -- fix a confirmed, major modeling error:
the marker-based pipeline's "unscaled" templates
(code/assets/models/unscaled/shoe_size_{5,7}_{NOEXO,EXO}.osim) are the
Rajagopal et al. 2016 model (<credits>Rajagopal, A, Dembia, C.L., ...
</credits>), NOT LaiUhlrich2022 as every docstring/note in this project
claimed -- confirmed by directly inspecting the raw XML's <credits>/
<publications> tags. No LaiUhlrich2022 base model has ever existed in this
project's own assets.

This script builds the CORRECT unscaled model: a genuine generic
LaiUhlrich2022.osim (sourced from the local opencap-processing install,
credits confirm "Rajagopal et al. (2016), Lai et al. (2017), Uhlrich et
al. (2022)" -- the real lineage OpenCap itself uses) with this study's own
57-marker ("Biomech57") markerset applied on top, exactly as intended.

The 57-marker set is lifted verbatim (same names, same LOCAL parent-body
offsets) from the current (wrong-skeleton) Rajagopal-based template --
LaiUhlrich2022 shares Rajagopal's underlying body segment geometry/local
frames for every body these markers attach to except one: LaiUhlrich2022
has no separate "head" body (OpenCap's video-based model doesn't track
head pose the way marker mocap can), so 4 markers (RAH, LAH, LPH, RPH)
have no valid parent -- per the user's explicit instruction, these 4 are
DROPPED rather than reattached elsewhere. Final markerset: 53 markers.

Run from code/, in the opensimPy environment:
    python expt_temp/markerbasedMODEL/build_lai_biomech57_model.py
"""
import re
import shutil
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from opensim_env import opensim_module  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
LAI_SOURCE = Path(r"C:\Users\sjogleka\Documents\opencap-processing\OpenSimPipeline\Models\LaiUhlrich2022.osim")
CURRENT_WRONG_TEMPLATE = CODE_ROOT / "assets" / "models" / "unscaled" / "shoe_size_7_NOEXO.osim"
OUTPUT_MODEL = OUT_DIR / "LaiUhlrich2022_Biomech57_unscaled.osim"

DROPPED_MARKERS = {"RAH", "LAH", "LPH", "RPH"}  # no "head" body in LaiUhlrich2022 -- dropped per user's instruction


def _extract_marker_blocks(template_text: str) -> list[str]:
    blocks = re.findall(r"<Marker name=\"[^\"]+\">.*?</Marker>", template_text, re.S)
    if len(blocks) != 57:
        raise ValueError(f"Expected 57 markers in the current template, found {len(blocks)}")
    kept = [b for b in blocks if re.match(r'<Marker name="([^"]+)"', b).group(1) not in DROPPED_MARKERS]
    if len(kept) != 57 - len(DROPPED_MARKERS):
        raise ValueError(f"Expected {57 - len(DROPPED_MARKERS)} markers after dropping head markers, found {len(kept)}")
    return kept


def build():
    if not LAI_SOURCE.is_file():
        raise FileNotFoundError(f"Generic LaiUhlrich2022 model not found at {LAI_SOURCE}")
    if not CURRENT_WRONG_TEMPLATE.is_file():
        raise FileNotFoundError(CURRENT_WRONG_TEMPLATE)

    lai_text = LAI_SOURCE.read_text(encoding="utf-8")
    credits = re.search(r"<credits>.*?</credits>", lai_text, re.S)
    print(f"Source model credits: {credits.group() if credits else 'MISSING'}")

    wrong_text = CURRENT_WRONG_TEMPLATE.read_text(encoding="utf-8")
    wrong_credits = re.search(r"<credits>.*?</credits>", wrong_text, re.S)
    print(f"Current (wrong) template credits: {wrong_credits.group() if wrong_credits else 'MISSING'}")

    marker_blocks = _extract_marker_blocks(wrong_text)
    print(f"Lifted {len(marker_blocks)} of 57 markers (dropped {sorted(DROPPED_MARKERS)}: no 'head' body in LaiUhlrich2022)")

    # Confirm every marker's parent body actually exists in the LaiUhlrich2022 skeleton
    # before writing anything -- fail loudly here, not silently at scale time.
    lai_bodies = set(re.findall(r'<Body name="([^"]+)"', lai_text))
    missing_bodies = set()
    for block in marker_blocks:
        parent = re.search(r"socket_parent_frame>/bodyset/([^/<]+)", block).group(1)
        if parent not in lai_bodies:
            missing_bodies.add(parent)
    if missing_bodies:
        raise ValueError(f"LaiUhlrich2022 is missing body/bodies referenced by markers: {missing_bodies}")
    print("All marker parent bodies confirmed present in LaiUhlrich2022's skeleton.")

    new_marker_set = (
        '<MarkerSet name="markerset">\n\t\t\t<objects>\n'
        + "\n".join(marker_blocks)
        + "\n\t\t\t</objects>\n\t\t\t<groups />\n\t\t</MarkerSet>"
    )
    output_text = re.sub(
        r"<MarkerSet name=\"markerset\">\s*<objects\s*/>\s*<groups\s*/>\s*</MarkerSet>",
        new_marker_set, lai_text, count=1,
    )
    if output_text == lai_text:
        raise ValueError("Empty <MarkerSet> placeholder not found in the source LaiUhlrich2022 model -- can't insert markers")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MODEL.write_text(output_text, encoding="utf-8")
    print(f"Wrote {OUTPUT_MODEL}")

    # Sanity-load through OpenSim itself -- if the XML is malformed or a
    # socket can't resolve, this raises immediately rather than failing
    # later, silently, inside ScaleTool.
    osim = opensim_module()
    model = osim.Model(str(OUTPUT_MODEL))
    model.finalizeConnections()
    marker_set = model.getMarkerSet()
    print(f"OpenSim load OK: {marker_set.getSize()} markers, {model.getBodySet().getSize()} bodies, "
          f"{model.getMuscles().getSize()} muscles, {model.getCoordinateSet().getSize()} coordinates")


if __name__ == "__main__":
    build()
