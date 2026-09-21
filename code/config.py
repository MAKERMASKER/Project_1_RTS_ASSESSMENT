"""One configuration object for the complete RTS assessment project.

Single source of truth: every path is a property derived from ``data_root``.
No legacy uppercase module-level aliases -- always go through a ``Config``
instance (or the ``DEFAULT`` singleton below).
"""
from dataclasses import dataclass, field
from pathlib import Path
import os


def _default_data_root() -> Path:
    override = os.environ.get("RTS_DATA_ROOT")
    if override:
        return Path(override)
    # code/config.py -> parents[0] = code/, parents[1] = repo root
    return Path(__file__).resolve().parents[1] / "data"


@dataclass(frozen=True)
class Config:
    """All paths and processing constants in one visible place."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    data_root: Path = field(default_factory=_default_data_root)

    conditions: tuple = ("NOEXO", "EXO")
    activity: str = "single_leg_jump"
    token: str = "SLJ"

    contact_force_n: float = 100.0
    active_force_plate_n: float = 20.0
    min_contact_duration_s: float = 0.020
    gravity_mps2: float = 9.80665
    com_smoothing_alpha: float = 160.0
    id_lowpass_hz: float = 4.5
    # np.gradient falls back from a central to a one-sided (lower-order)
    # difference at an array's first/last sample; double-differentiating COM
    # for GRF compounds that boundary error badly. Computing over a window
    # padded by this many seconds on each side (then cropping back to the
    # reported window) keeps that error out of the actual output -- see
    # framework_processor.py's _padded_time().
    com_gradient_pad_s: float = 0.15
    jump_window_s: tuple = (-0.75, 0.75)
    opensim_log_level: str = "error"
    overwrite: bool = False
    c3d_rotation_axis: str = "x"
    c3d_rotation_degrees: float = -90.0
    open_foot_coordinates: tuple = (
        "subtalar_angle_r", "subtalar_angle_l", "mtp_angle_r", "mtp_angle_l",
    )
    ik_marker_weight: float = 1.0

    # LSI comparisons: (name, test_condition, test_side, reference_condition, reference_side).
    # Verified against Isokinetic_Data.xlsx: PEAK_TORQUE_NOEXO_LSI == 100*LEFT_NOEXO/RIGHT_NOEXO,
    # PEAK_TORQUE_WITHEXO_LSI == 100*LEFT_NOEXO/RIGHT_WITHEXO.
    lsi_definitions: tuple = (
        ("NOEXO_L_vs_R", "NOEXO", "L", "NOEXO", "R"),
        ("NOEXO_L_vs_EXO_R", "NOEXO", "L", "EXO", "R"),
    )
    # Movement-task condition label -> Isokinetic_Data.xlsx condition label.
    isokinetic_condition_map: dict = field(
        default_factory=lambda: {"NOEXO": "NOEXO", "EXO": "WITHEXO"}
    )

    @property
    def subject_rules_root(self) -> Path:
        return self.project_root / "subject_rules"

    @property
    def results_root(self) -> Path:
        return self.project_root.parent / "results"

    @property
    def assets_root(self) -> Path:
        return self.project_root / "assets"

    @property
    def models_root(self) -> Path:
        return self.assets_root / "models"

    @property
    def scale_templates_root(self) -> Path:
        return self.assets_root / "setup_templates" / "scale"

    @property
    def opencap_ik_setup_template(self) -> Path:
        return self.assets_root / "opencap_ik" / "Setup_IK_OpenCap_official.xml"

    @property
    def calibration_root(self) -> Path:
        return self.results_root / "calibration"

    @property
    def preflight_root(self) -> Path:
        return self.results_root / "_preflight"

    @property
    def metadata_file(self) -> Path:
        return self.data_root / "subject_metadata.csv"

    @property
    def session_file(self) -> Path:
        return self.data_root / "opencap_sessions.csv"

    @property
    def trial_status_file(self) -> Path:
        return self.data_root / "trial_status.csv"

    @property
    def isokinetic_file(self) -> Path:
        return self.data_root / "Isokinetic_Data.xlsx"


DEFAULT = Config()
