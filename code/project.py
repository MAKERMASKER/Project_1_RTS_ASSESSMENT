"""RtsAssessmentProject: the single facade wiring one subject to every processor."""
from activities import activity_for
from calibration import InsoleCalibration
from config import DEFAULT
from correlation import CorrelationAnalysis
from discovery import TrialDiscovery
from framework_processor import FrameworkProcessor
from lsi import LsiCalculator
from marker_processor import MarkerProcessor
from outcomes import OutcomeExtractor
from preflight import Preflight
from subject import Subject


class RtsAssessmentProject:
    def __init__(self, subject_id: str, config=DEFAULT):
        self.config = config
        self.subject = Subject(subject_id, config)
        self.discovery = TrialDiscovery(config)
        self.marker_processor = MarkerProcessor(config)
        self.calibration = InsoleCalibration(config)
        self.framework_processor = FrameworkProcessor(config)
        self.outcome_extractor = OutcomeExtractor(config)
        self.lsi_calculator = LsiCalculator(config)
        self.correlation = CorrelationAnalysis(config)

    def activity(self, name: str = "single_leg_jump"):
        return activity_for(name, self.config)

    def preflight(self, name: str = "single_leg_jump"):
        return Preflight(self.config).run(self.activity(name), subject_ids=[self.subject.id])

    def run_marker(self, name: str = "single_leg_jump", condition: str = "all", convert_c3d: bool = True):
        return self.marker_processor.run(self.subject, self.activity(name), condition, convert_c3d)

    def run_calibration(self):
        return self.calibration.fit(self.subject)

    def run_framework(self, name: str = "single_leg_jump", condition: str = "all", trial_name: str = None):
        return self.framework_processor.run(self.subject, self.activity(name), condition, trial_name)

    def run_analysis(self, name: str = "single_leg_jump", make_plots: bool = True):
        activity = self.activity(name)
        summary = self.outcome_extractor.run(self.subject, activity, make_plots)
        ratios = self.lsi_calculator.run(self.subject, activity, summary, make_plots)
        self.correlation.subject_report(self.subject, activity, make_plots)
        return ratios

    def describe(self, name: str = "single_leg_jump"):
        return self.activity(name).describe()
