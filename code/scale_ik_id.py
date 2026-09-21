"""ModelScaler / InverseKinematicsRunner / InverseDynamicsRunner: OpenSim setup-XML
generation and tool invocation for the marker-based (ground-truth) pipeline.

OpenSim's Python API has no friendlier setup-file builder than XML text, so
setup XML is still built via string substitution here -- kept isolated to
this one module so it's easy to find and test.
"""
from pathlib import Path
import os
import re
import shutil

import numpy as np
import pandas as pd

from config import DEFAULT
from io_tables import read_trc, read_storage
from opensim_env import opensim_module


def xml_path(path) -> str:
    return Path(path).resolve().as_posix().replace("&", "&amp;")


def _relative(path, folder) -> str:
    return os.path.relpath(Path(path), Path(folder)).replace("\\", "/")


def _replace_tag(text: str, tag: str, value) -> str:
    return re.sub(rf"<{tag}>\s*[^<]*\s*</{tag}>", f"<{tag}>{value}</{tag}>", text)


class ModelScaler:
    """Scale the unscaled shoe-size/condition model to one subject."""

    def __init__(self, config=DEFAULT):
        self.config = config

    def unscaled_model_path(self, shoe_size: int, condition: str) -> Path:
        if shoe_size not in (5, 7):
            raise ValueError(f"shoe_size must be 5 or 7; found {shoe_size}")
        return self.config.models_root / "unscaled" / f"shoe_size_{shoe_size}_{condition.upper()}.osim"

    def _models_folder(self, subject) -> Path:
        return self.config.results_root / subject.id / "marker_based" / "models"

    def scale(self, subject, condition: str) -> Path:
        """Scale the model for one subject/condition; returns the scaled model path."""
        osim = opensim_module()
        condition = condition.upper()
        size = subject.shoe_size
        mass = subject.mass_kg
        height = subject.height_m
        if size is None or mass is None or height is None:
            raise ValueError(f"{subject.id} is missing ShoeSize/Mass_kg/Height_m metadata")

        folder = self._models_folder(subject)
        folder.mkdir(parents=True, exist_ok=True)
        anatomy = folder / f"{subject.id}_shoe_size_{size}_anatomical.osim"
        output = folder / f"{subject.id}_shoe_size_{size}_{condition}_scaled.osim"
        factors = folder / f"{subject.id}_scale_factors.xml"
        noexo_static = (
            subject.data_folder / "PROCESSED" / "MOCAP" / "marker_data"
            / f"{subject.id}_NOEXO_static_markers_PROCESSED.trc"
        )

        anatomical_setup = folder / "Setup_Scale_NOEXO.xml"
        self._write_scale_setup(
            self.config.scale_templates_root / "Scale_NOEXO.xml",
            anatomical_setup, self.unscaled_model_path(size, "NOEXO"), noexo_static,
            anatomy, factors, mass, height,
        )
        if self.config.overwrite or not anatomy.is_file():
            if not osim.ScaleTool(str(anatomical_setup)).run():
                raise RuntimeError(f"Anatomical scaling failed for {subject.id}")

        if condition == "NOEXO":
            shutil.copy2(anatomy, output)
        else:
            condition_static = (
                subject.data_folder / "PROCESSED" / "MOCAP" / "marker_data"
                / f"{subject.id}_{condition}_static_markers_PROCESSED.trc"
            )
            condition_setup = folder / f"Setup_Scale_{condition}.xml"
            self._write_scale_setup(
                self.config.scale_templates_root / f"Scale_{condition}.xml",
                condition_setup, self.unscaled_model_path(size, condition), condition_static,
                output, factors, mass, height, manual=True,
            )
            if self.config.overwrite or not output.is_file():
                if not osim.ScaleTool(str(condition_setup)).run():
                    raise RuntimeError(f"{condition} marker placement failed for {subject.id}")

        self._unlock_coordinates(anatomy)
        self._unlock_coordinates(output)
        return output

    def prepare_lai_model(self, subject, condition: str, source_opencap_model, overwrite: bool = False) -> Path:
        """A single, shared, mass-corrected model for BOTH the marker-based and
        framework pipelines.

        Starts from the subject's own OpenCap-scaled model for this condition
        (`source_opencap_model` -- consistent body-segment scaling across every
        trial), corrects total body mass (and each body's mass/inertia
        proportionally) to exactly match the subject's *measured* mass (OpenCap's
        own video-based mass estimate does not always match it, and that error
        would otherwise propagate into every downstream COM-based force
        estimate), then replaces its markers with the study's own reference
        markers wherever they also appear in this subject's static
        motion-capture trial. Using the *same* model for marker-based IK/ID and
        framework GRF/ID means any remaining framework-vs-marker discrepancy
        reflects the measurement modality, not two different anatomical models.
        """
        osim = opensim_module()
        condition = condition.upper()
        folder = self._models_folder(subject)
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / f"{subject.id}_{condition}_LaiUhlrich2022.osim"
        if output.is_file() and not overwrite:
            return output

        static = (
            subject.data_folder / "PROCESSED" / "MOCAP" / "marker_data"
            / f"{subject.id}_{condition}_static_markers_PROCESSED.trc"
        )
        reference = self.unscaled_model_path(subject.shoe_size, condition)
        mass = subject.mass_kg
        if mass is None:
            raise ValueError(f"{subject.id} is missing Mass_kg metadata")

        model = osim.Model(str(source_opencap_model))
        old_mass = sum(model.getBodySet().get(i).getMass() for i in range(model.getBodySet().getSize()))
        ratio = mass / old_mass
        for index in range(model.getBodySet().getSize()):
            body = model.updBodySet().get(index)
            body.setMass(body.getMass() * ratio)
            inertia = body.getInertia()
            body.setInertia(osim.Inertia(
                osim.Vec3(*(self._vector_values(inertia.getMoments()) * ratio)),
                osim.Vec3(*(self._vector_values(inertia.getProducts()) * ratio)),
            ))

        model.updMarkerSet().clearAndDestroy()
        source = osim.Model(str(reference))
        static_names = set(read_trc(static)["marker_names"])
        marker_names = []
        for index in range(source.getMarkerSet().getSize()):
            source_marker = source.getMarkerSet().get(index)
            name = source_marker.getName()
            body_name = source_marker.getParentFrame().getName()
            if name not in static_names or not model.getBodySet().contains(body_name):
                continue
            model.addMarker(osim.Marker(
                name, model.getBodySet().get(body_name), source_marker.get_location()
            ))
            marker_names.append(name)
        if not marker_names:
            raise RuntimeError(f"No study markers shared by {reference} and {static}")

        setup_folder = folder / "setup"
        setup_folder.mkdir(exist_ok=True)
        seed = setup_folder / f"{subject.id}_{condition}_Lai_seed.osim"
        model.finalizeConnections()
        model.printToXML(str(seed))

        tool = osim.ScaleTool()
        tool.setSubjectMass(mass)
        tool.getGenericModelMaker().setModelFileName(str(seed))
        tool.getModelScaler().setApply(False)
        placer = tool.getMarkerPlacer()
        placer.setApply(True)
        placer.setMarkerFileName(str(static))
        placer.setMoveModelMarkers(True)
        placer.setOutputModelFileName(str(output))
        placer.setOutputMotionFileName(str(setup_folder / f"{condition}_static_pose.mot"))
        placer.setOutputMarkerFileName(str(setup_folder / f"{condition}_placed_markers.xml"))
        time = read_trc(static)["time"]
        limits = osim.ArrayDouble()
        limits.append(float(time[0]))
        limits.append(float(time[-1]))
        placer.setTimeRange(limits)
        tasks = placer.getIKTaskSet()
        tasks.clearAndDestroy()
        for name in marker_names:
            task = osim.IKMarkerTask()
            task.setName(name)
            task.setApply(True)
            task.setWeight(1.0)
            tasks.cloneAndAppend(task)
        tool.printToXML(str(setup_folder / f"Setup_Lai_{condition}.xml"))
        if not tool.run():
            raise RuntimeError(f"Lai marker placement failed: {subject.id} {condition}")

        final = osim.Model(str(output))
        final_mass = sum(final.getBodySet().get(i).getMass() for i in range(final.getBodySet().getSize()))
        if abs(final_mass - mass) > 1e-5:
            raise RuntimeError(f"Final model mass {final_mass} does not match metadata {mass}")
        return output

    @staticmethod
    def _vector_values(vector):
        return np.array([vector.get(i) for i in range(3)])

    def _write_scale_setup(self, template, setup, model, markers, output, factors,
                            mass, height_m, manual=False):
        for path in (template, model, markers):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        setup = Path(setup)
        setup.parent.mkdir(parents=True, exist_ok=True)
        trc = read_trc(markers)
        text = Path(template).read_text()
        values = {
            "model_file": _relative(model, setup.parent),
            "marker_file": _relative(markers, setup.parent),
            "output_model_file": _relative(output, setup.parent),
            "output_scale_file": _relative(factors, setup.parent),
            "output_motion_file": "StaticPose.mot",
            "output_marker_file": "PlacedMarkers.xml",
            "time_range": f"{np.nanmin(trc['time']):.8f} {np.nanmax(trc['time']):.8f}",
            "mass": f"{mass:.15g}",
            "height": f"{1000 * height_m:.15g}",
        }
        for tag, value in values.items():
            text = _replace_tag(text, tag, value)
        if manual and Path(factors).is_file():
            text = self._manual_factors(text, factors)
        setup.write_text(text)
        return setup

    @staticmethod
    def _manual_factors(text: str, factor_file) -> str:
        source = Path(factor_file).read_text()
        match = re.search(r"<objects>\s*([\s\S]*?)\s*</objects>", source)
        if not match:
            raise ValueError(f"No scale factors in {factor_file}")
        text = re.sub(
            r"<scaling_order>\s*[^<]*\s*</scaling_order>",
            "<scaling_order>manualScale</scaling_order>", text, count=1,
        )
        text = re.sub(
            r"<MeasurementSet>\s*<objects>[\s\S]*?</objects>\s*<groups\s*/>\s*</MeasurementSet>",
            "<MeasurementSet><objects/><groups/></MeasurementSet>", text, count=1,
        )
        return re.sub(
            r"<ScaleSet>\s*(?:<objects\s*/>|<objects>[\s\S]*?</objects>)\s*<groups\s*/>\s*</ScaleSet>",
            f"<ScaleSet><objects>{match.group(1)}</objects><groups/></ScaleSet>", text, count=1,
        )

    def _unlock_coordinates(self, model_path):
        osim = opensim_module()
        model = osim.Model(str(model_path))
        coordinates = model.updCoordinateSet()
        for name in self.config.open_foot_coordinates:
            if coordinates.contains(name):
                coordinate = coordinates.get(name)
                coordinate.setDefaultLocked(False)
                coordinate.set_locked(False)
        model.printToXML(str(model_path))


class InverseKinematicsRunner:
    def __init__(self, config=DEFAULT):
        self.config = config

    @staticmethod
    def _shared_markers(trc_path, model_path) -> list[str]:
        trc = read_trc(trc_path)
        model_names = re.findall(r'<Marker\s+name="([^"]+)"', Path(model_path).read_text(errors="replace"))
        lookup = {name.lower(): name for name in model_names}
        return [lookup[name.lower()] for name in trc["marker_names"] if name.lower() in lookup]

    def run(self, subject, trial, model_path: Path):
        if not trial.marker_trc.is_file():
            return None
        osim = opensim_module()
        out = self.config.results_root / subject.id / "marker_based" / trial.condition / "ik"
        out.mkdir(parents=True, exist_ok=True)
        motion = out / f"{trial.name}_ik.mot"
        if motion.is_file() and not self.config.overwrite:
            return motion

        tasks = "".join(
            f'<IKMarkerTask name="{marker}"><apply>true</apply>'
            f'<weight>{self.config.ik_marker_weight:.8g}</weight></IKMarkerTask>'
            for marker in self._shared_markers(trial.marker_trc, model_path)
        )
        time = read_trc(trial.marker_trc)["time"]
        setup = out / f"Setup_IK_{trial.name}.xml"
        setup.write_text(
            f'<?xml version="1.0"?><OpenSimDocument Version="40000"><InverseKinematicsTool>'
            f'<model_file>{xml_path(model_path)}</model_file><constraint_weight>Inf</constraint_weight>'
            f'<accuracy>1e-5</accuracy><IKTaskSet><objects>{tasks}</objects><groups/></IKTaskSet>'
            f'<marker_file>{xml_path(trial.marker_trc)}</marker_file><coordinate_file>Unassigned</coordinate_file>'
            f'<time_range>{np.nanmin(time):.8f} {np.nanmax(time):.8f}</time_range>'
            f'<report_errors>true</report_errors><output_motion_file>{xml_path(motion)}</output_motion_file>'
            f'</InverseKinematicsTool></OpenSimDocument>'
        )
        osim.InverseKinematicsTool(str(setup)).run()
        return motion


class InverseDynamicsRunner:
    def __init__(self, config=DEFAULT):
        self.config = config

    @staticmethod
    def _external_force(name, body, index) -> str:
        return (
            f'<ExternalForce name="{name}"><applied_to_body>{body}</applied_to_body>'
            f'<force_expressed_in_body>ground</force_expressed_in_body>'
            f'<point_expressed_in_body>ground</point_expressed_in_body>'
            f'<force_identifier>ground_force_{index}_v</force_identifier>'
            f'<point_identifier>ground_force_{index}_p</point_identifier>'
            f'<torque_identifier>ground_moment_{index}_m</torque_identifier></ExternalForce>'
        )

    def run(self, subject, trial, model_path: Path):
        """Right force is always ground_force_1 -> calcn_r; left is force_2 -> calcn_l."""
        ik = self.config.results_root / subject.id / "marker_based" / trial.condition / "ik" / f"{trial.name}_ik.mot"
        if not ik.is_file() or not trial.force_mot.is_file():
            return None
        out = self.config.results_root / subject.id / "marker_based" / trial.condition / "id"
        out.mkdir(parents=True, exist_ok=True)
        result = out / f"{trial.name}_inverse_dynamics.sto"
        if result.is_file() and not self.config.overwrite:
            return result

        ik_data = read_storage(ik)
        force_data = read_storage(trial.force_mot)
        start = max(ik_data.time.min(), force_data.time.min())
        stop = min(ik_data.time.max(), force_data.time.max())
        setup_folder = out / f"setup_{trial.name}"
        return self.run_with_paths(model_path, ik, trial.force_mot, result, start, stop, setup_folder)

    def run_with_paths(self, model_path, coordinates_path, grf_path, output_path: Path,
                        start=None, stop=None, setup_folder: Path = None):
        """Generic ID: build ExternalLoads/InverseDynamics setup XML and run.

        Right force is always ground_force_1 -> calcn_r; left is force_2 ->
        calcn_l -- the one bilateral-force convention used by both the
        marker-based and framework pipelines, so both can call this directly.
        """
        osim = opensim_module()
        output_path = Path(output_path)
        setup_folder = Path(setup_folder) if setup_folder else output_path.parent / f"setup_{output_path.stem}"
        setup_folder.mkdir(parents=True, exist_ok=True)

        loads = setup_folder / "ExternalLoads.xml"
        loads.write_text(
            f'<?xml version="1.0"?><OpenSimDocument Version="40000"><ExternalLoads><objects>'
            f'{self._external_force("right", "calcn_r", 1)}{self._external_force("left", "calcn_l", 2)}'
            f'</objects><groups/><datafile>{xml_path(grf_path)}</datafile></ExternalLoads></OpenSimDocument>'
        )

        coordinates = read_storage(coordinates_path)
        if start is None:
            start = coordinates.time.min()
        if stop is None:
            stop = coordinates.time.max()
        setup = setup_folder / "InverseDynamics.xml"
        setup.write_text(
            f'<?xml version="1.0"?><OpenSimDocument Version="40000"><InverseDynamicsTool>'
            f'<results_directory>{xml_path(output_path.parent)}</results_directory>'
            f'<model_file>{xml_path(model_path)}</model_file>'
            f'<time_range>{start:.8f} {stop:.8f}</time_range><forces_to_exclude>Muscles</forces_to_exclude>'
            f'<external_loads_file>{xml_path(loads)}</external_loads_file>'
            f'<coordinates_file>{xml_path(coordinates_path)}</coordinates_file>'
            f'<lowpass_cutoff_frequency_for_coordinates>{self.config.id_lowpass_hz}</lowpass_cutoff_frequency_for_coordinates>'
            f'<output_gen_force_file>{output_path.name}</output_gen_force_file>'
            f'</InverseDynamicsTool></OpenSimDocument>'
        )
        if not osim.InverseDynamicsTool(str(setup)).run():
            raise RuntimeError(f"Inverse dynamics failed: {output_path.name}")
        return output_path
