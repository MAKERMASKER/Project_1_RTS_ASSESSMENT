"""MarkerProcessor: orchestrates the ground-truth pipeline for one subject.

C3D conversion -> force-plate resultants -> independent anatomical scaling
-> IK -> ID. Delegates every step to its own class; this file only sequences
them and only ever processes trials TrialDiscovery has already classified
as usable.

Uses ModelScaler.scale() -- the marker-based (ground-truth) pipeline scales
its OWN LaiUhlrich2022 model (with the study's Biomech57 markerset) from
each subject's own NOEXO static trial via real measurement-based scaling;
the EXO condition reuses NOEXO's own scale factors (manualScale) and only
reruns MarkerPlacer on the EXO static trial to reposition markers, never
rescaling. This is deliberately independent of whatever model OpenCap's own
video-based pipeline produces (see framework_processor.py, which uses
`trial.opencap_model` directly, untouched) -- sharing one model between the
two pipelines would defeat the purpose of comparing them as two genuinely
independent measurement modalities.
"""
from config import DEFAULT
from c3d_convert import C3dConverter
from discovery import TrialDiscovery
from force_plates import ForcePlateProcessor
from scale_ik_id import InverseDynamicsRunner, InverseKinematicsRunner, ModelScaler


class MarkerProcessor:
    def __init__(self, config=DEFAULT):
        self.config = config
        self.c3d_converter = C3dConverter(config)
        self.force_plates = ForcePlateProcessor(config)
        self.scaler = ModelScaler(config)
        self.ik_runner = InverseKinematicsRunner(config)
        self.id_runner = InverseDynamicsRunner(config)
        self.discovery = TrialDiscovery(config)

    def run(self, subject, activity, condition: str = "all", convert_c3d: bool = True):
        conditions = self.config.conditions if condition.lower() == "all" else (condition.upper(),)
        completed = []
        for item in conditions:
            if convert_c3d:
                print(f"{subject.id} {item}: converting C3D markers and forces")
                self.c3d_converter.convert(subject, item)

            print(f"{subject.id} {item}: scaling anatomical model")
            model = self.scaler.scale(subject, item)

            for trial in self.discovery.trials(subject, activity, item):
                print(f"{trial.name}: resultant forces -> IK -> ID")
                try:
                    self.force_plates.process(activity, trial)
                    self.ik_runner.run(subject, trial, model)
                    self.id_runner.run(subject, trial, model)
                except FileNotFoundError as error:
                    # A trial that passed name/exclusion classification but is
                    # missing an actual input file (e.g. its C3D failed to
                    # produce a force file) must not abort the whole batch.
                    print(f"  skipped ({error})")
                    continue
                completed.append(trial)
        return completed
