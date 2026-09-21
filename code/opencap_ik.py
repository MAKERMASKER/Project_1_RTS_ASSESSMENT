"""OpenCapIkRunner: rerun OpenCap's own official IK setup locally.

Needed whenever a TRC is used that OpenCap's cloud pipeline never saw. The
main case: once a TRC is repaired (trc_repair.py), OpenCap's own
`OpenSimData/Kinematics/*.mot` still reflects the ORIGINAL, unrepaired TRC --
it's now stale. The pipeline must regenerate matching IK output from the
repaired TRC using the same model/marker-weight setup OpenCap itself uses
(its own official IK task weights, with the patella bodies/joints/
constraints removed exactly as OpenCap's own IK does), not a different
recipe -- otherwise "repaired vs. original" comparisons and downstream use
would be confounded by a second, unrelated methodology change.

Standalone (no dependency on framework_processor.py or trc_repair.py) so
both can use it without a circular import: trc_repair.py's
TrcRepairValidator uses it for matched original-vs-repaired comparison;
framework_processor.py uses it to actually regenerate production IK for a
promoted repair.
"""
from pathlib import Path

from config import DEFAULT
from io_tables import read_trc
from opensim_env import opensim_module


class OpenCapIkRunner:
    def __init__(self, config=DEFAULT):
        self.config = config

    def _opencap_equivalent_model(self, source_model_path, out_path: Path) -> Path:
        """OpenCap's own IK model: no muscles, no patella bodies/joints/constraints."""
        osim = opensim_module()
        model = osim.Model(str(source_model_path))
        model.getForceSet().setSize(0)
        for name in ("patellofemoral_knee_angle_r_con", "patellofemoral_knee_angle_l_con"):
            index = model.getConstraintSet().getIndex(name, 0)
            if index >= 0:
                model.getConstraintSet().remove(index)
        for name in ("patella_r", "patella_l"):
            index = model.getBodySet().getIndex(name, 0)
            if index >= 0:
                model.getBodySet().remove(index)
        for name in ("patellofemoral_r", "patellofemoral_l"):
            index = model.getJointSet().getIndex(name, 0)
            if index >= 0:
                model.getJointSet().remove(index)
        model.initSystem()
        model.printToXML(str(out_path))
        return out_path

    def run(self, trc_path, model_path, output_path, setup_folder: Path = None) -> Path:
        """Rerun OpenCap's own official IK setup: trc_path against model_path."""
        osim = opensim_module()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        setup_folder = Path(setup_folder) if setup_folder else output_path.parent / "setup"
        setup_folder.mkdir(parents=True, exist_ok=True)

        ik_model = setup_folder / "opencap_ik_model_no_patella.osim"
        self._opencap_equivalent_model(model_path, ik_model)

        time = read_trc(trc_path)["time"]
        tool = osim.InverseKinematicsTool(str(self.config.opencap_ik_setup_template))
        tool.setName(output_path.stem)
        tool.set_model_file(str(ik_model))
        tool.set_marker_file(str(trc_path))
        tool.set_time_range(0, float(time.min()))
        tool.set_time_range(1, float(time.max()))
        tool.setResultsDir(str(output_path.parent))
        tool.set_report_errors(True)
        tool.set_report_marker_locations(False)
        tool.set_output_motion_file(str(output_path))
        tool.printToXML(str(setup_folder / "Setup_OpenCap_Equivalent_IK.xml"))
        if not tool.run():
            raise RuntimeError(f"OpenCap-equivalent IK failed for {trc_path}")
        return output_path
