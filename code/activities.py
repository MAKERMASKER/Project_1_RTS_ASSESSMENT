"""Activity: the abstract base class every movement task implements.

Only SingleLegJump exists today. A second activity (triple hop, skater hop,
...) is added as a new Activity subclass following this one as the template
-- ForcePlateProcessor and FrameworkProcessor never change; they only ever
call trial.activity.marker_plates(...) / .synchronize(...) / .outcomes(...).

marker_plates() and synchronize() are activity-dependent by design (each
activity defines its own landing/contact detection and force-plate
assignment). Nothing here is subject- or calibration-dependent -- rules
(exclusions, per-trial overrides) are passed in explicitly by the caller,
never read from a stored Subject reference, so an Activity instance has no
per-subject state and can be reused across every subject.
"""
from abc import ABC, abstractmethod
import numpy as np
from scipy.signal import find_peaks

import io_tables
from config import DEFAULT
from insole_geometry import InsoleGeometry


def moving_mean(values, width):
    return np.convolve(np.asarray(values, dtype=float), np.ones(width) / width, mode="same")


def baseline_height(height):
    count = len(height)
    return np.nanmedian(height[max(0, round(0.05 * count) - 1):max(5, round(0.25 * count))])


def foot_landings(height, high, low, minimum_samples=8):
    """Indices where height rises above `high` then settles back below `low`."""
    landings = []
    in_flight = False
    below_count = 0
    for index, value in enumerate(height):
        if not in_flight and value >= high:
            in_flight = True
            below_count = 0
        elif in_flight and value <= low:
            below_count += 1
            if below_count == minimum_samples:
                landings.append(index - minimum_samples + 1)
                in_flight = False
        else:
            below_count = 0
    return landings


class Activity(ABC):
    """Shared per-trial workflow plus the two activity-specific rule methods."""

    name = ""
    title = ""
    token = ""
    unilateral_framework = False
    framework_interval = ""
    functional_outcomes = ()
    # Whether this activity's filenames carry a _L_/_R_ side segment
    # (<Subject>_<Condition>_<Token>_<Side>_<TrialNumber>). False for
    # activities like Walk where a single trial exercises both feet and the
    # raw filenames have no side segment at all (see discovery.py's
    # classify() and subject.py's side_from_name()).
    has_side_token = True

    def __init__(self, config=DEFAULT):
        self.config = config

    def contact(self, force):
        """The one transparent framework contact definition: force >= threshold."""
        return np.asarray(force) >= self.config.contact_force_n

    @abstractmethod
    def marker_plates(self, trial):
        """Which force plate(s) belong to the right/left foot for this trial.

        Returns (right_plate_numbers, left_plate_numbers), or None if plate
        assignment must be inferred dynamically (no activity needs that yet).
        """

    @abstractmethod
    def synchronize(self, trial, trc, ik_all, insole_all, rules):
        """Align insole time to marker/OpenCap time; return the analysis window.

        Returns (time, insole, ik, sync_info) where sync_info carries at
        least shift_s, peak_grf_time_s, quality_omit, method.
        """

    def contact_masks(self, trial, plate_forces: dict, dt: float):
        """Optional extra per-plate sample filter for ForcePlateProcessor's
        fixed-plate-list branch, applied on top of its own force threshold.

        `plate_forces` is {plate_number: vertical_force_array} for exactly
        the plates marker_plates() assigned to one side of this trial;
        return {plate_number: bool_array} to keep only some samples toward
        that side's resultant, or None (the default, used by every
        activity except LateralHop) for no extra filtering -- a pure
        no-op, so this is an additive hook, not a behavior change for
        existing activities.
        """
        return None

    def outcomes(self, trial, trc, sync):
        """Activity-specific scalar outcomes beyond the shared GRF/moment set.

        Returns (distance_m, count) -- NaN/0 when not applicable to this
        activity (SLJ has neither; triple hop uses distance, lateral hop
        uses count).
        """
        return np.nan, 0

    def describe(self):
        return {
            "activity": self.name,
            "token": self.token,
            "unilateral_framework": self.unilateral_framework,
            "framework_interval": self.framework_interval,
            "functional_outcomes": self.functional_outcomes,
        }

    def _selected_force(self, trial, insole):
        word = "right" if trial.side == "R" else "left"
        return insole[f"{word}_force"].to_numpy(float)

    def _force_contacts(self, force, minimum_seconds=0.0, time=None):
        contacts = io_tables.segments(force >= self.config.contact_force_n)
        if minimum_seconds and len(contacts):
            dt = float(np.median(np.diff(time)))
            duration = (contacts[:, 1] - contacts[:, 0] + 1) * dt
            contacts = contacts[duration >= minimum_seconds]
        return contacts

    def _finish_sync(
        self, trial, trc, ik_all, insole_all, open_event, insole_event,
        peak_index, method, open_landing_index=0, quality_omit=False,
        contact_only=False, discard_start_seconds=0.0, max_duration_seconds=None,
        peak_window_seconds=None,
    ):
        shift = float(open_event - insole_event)
        shifted = insole_all.time.to_numpy(float) + shift
        common = (
            (shifted >= max(trc["time"].min(), ik_all.time.min()))
            & (shifted <= min(trc["time"].max(), ik_all.time.max()))
        )
        selected_force = self._selected_force(trial, insole_all)
        if peak_window_seconds is not None:
            peak_time = shifted[peak_index]
            common &= shifted >= peak_time + peak_window_seconds[0]
            common &= shifted <= peak_time + peak_window_seconds[1]
        if contact_only:
            contacts = self._force_contacts(selected_force)
            selected = contacts[(contacts[:, 0] <= peak_index) & (contacts[:, 1] >= peak_index)]
            if not len(selected):
                raise ValueError(f"Selected contact was not found for {trial.name}")
            interval = np.zeros(len(selected_force), dtype=bool)
            interval[selected[0, 0]:selected[0, 1] + 1] = True
            common &= interval
        if discard_start_seconds:
            interval_start = trc["time"].min() + discard_start_seconds
            common &= shifted >= interval_start
            if max_duration_seconds is not None:
                common &= shifted <= interval_start + max_duration_seconds
        indices = np.flatnonzero(common)
        if not len(indices):
            raise ValueError(f"No synchronized overlap for {trial.name}")
        use = slice(indices[0], indices[-1] + 1)
        time = shifted[use]
        return (
            time,
            insole_all.iloc[use].reset_index(drop=True),
            io_tables.interpolate_table(ik_all, time),
            {
                "shift_s": shift,
                "peak_grf_time_s": shifted[peak_index],
                "quality_omit": quality_omit,
                "open_landing_index": open_landing_index,
                "method": method,
            },
        )


class SingleLegJump(Activity):
    """Complete jump: 0.75 s before to 0.75 s after the landing GRF peak."""

    name = "single_leg_jump"
    title = "Single-leg jump"
    token = "SLJ"
    unilateral_framework = True
    framework_interval = "complete jump: 0.75 s before to 0.75 s after landing GRF peak"
    functional_outcomes = ("GRF", "COP", "joint moments", "knee work", "LSI")

    def marker_plates(self, trial):
        return ([2], []) if trial.side == "R" else ([], [6])

    def synchronize(self, trial, trc, ik_all, insole_all, rules):
        force = moving_mean(self._selected_force(trial, insole_all), 7)
        height = moving_mean(InsoleGeometry.foot_height(trc, trial.side), 9)
        baseline = baseline_height(height)

        peaks = np.flatnonzero(
            (height[1:-1] >= height[:-2]) & (height[1:-1] >= height[2:])
            & (height[1:-1] >= baseline + 0.03)
        ) + 1
        if not len(peaks):
            raise ValueError(f"Single-leg jump flight was not found for {trial.name}")
        use_last = trial.name in rules["use_last_flight_peak"]
        flight_peak = peaks[-1] if use_last else peaks[np.argmax(height[peaks])]

        landings = io_tables.segments(
            (np.arange(len(height)) > flight_peak) & (height <= baseline + 0.02)
        )
        landings = landings[(landings[:, 1] - landings[:, 0] + 1) >= 8]

        contacts = self._force_contacts(force)
        if len(contacts):
            dt = float(np.median(np.diff(insole_all.time)))
            duration = (contacts[:, 1] - contacts[:, 0] + 1) * dt
            contacts = contacts[duration >= 0.10]
        if not len(landings) or not len(contacts):
            raise ValueError(f"SLJ landing/contact was not found for {trial.name}")

        choices = []
        for start, end in contacts:
            peak = start + int(np.argmax(force[start:end + 1]))
            choices.append((start, end, peak, force[peak]))
        ordinal = rules["landing_ordinal"].get(trial.name)
        chosen = choices[ordinal - 1] if ordinal else max(choices, key=lambda item: item[3])

        return self._finish_sync(
            trial, trc, ik_all, insole_all,
            trc["time"][landings[0, 0]], insole_all.time.iloc[chosen[0]], chosen[2],
            "complete_jump_around_landing_peak", landings[0, 0],
            peak_window_seconds=self.config.jump_window_s,
        )


class Walk(Activity):
    """Overground walking: multiple gait cycles, both feet active throughout.

    Force-plate assignment is fixed per the study protocol (not inferred):
    left foot steps on plates 3/1/6/5, right foot on plates 2/4, in walking
    order along the walkway. A foot spanning two plates simultaneously is
    handled by ForcePlateProcessor._resultant()'s existing per-sample
    force-weighted-average combining -- unchanged from what SingleLegJump
    already uses (SLJ is just the one-plate-per-foot case of the same
    formula), per the project's "one implementation of the multi-plate
    combining technique, reused by every activity" convention.

    Raw WALK filenames have no _L_/_R_ side segment (see
    discovery.py/subject.py's has_side_token handling) since a walking trial
    is inherently bilateral -- trial.side is None for Walk trials and
    marker_plates() does not consult it.

    No OpenCap markerless session exists for any WALK trial yet, so
    synchronize() (needed only by FrameworkProcessor/InsoleCalibration, not
    by the marker-based pipeline) is deliberately left unimplemented.
    """

    name = "walk"
    title = "Walking (overground gait)"
    token = "WALK"
    has_side_token = False
    unilateral_framework = False
    framework_interval = "not yet defined -- no OpenCap walking sessions exist yet"
    functional_outcomes = ()

    def marker_plates(self, trial):
        return ([2, 4], [3, 1, 6, 5])

    def synchronize(self, trial, trc, ik_all, insole_all, rules):
        raise NotImplementedError(
            "Walk.synchronize() is not implemented -- no OpenCap markerless "
            "session exists for any WALK trial yet, so there is nothing to "
            "synchronize insole time against. Marker-based processing "
            "(c3d convert -> force-plate combine -> scale/IK/ID) does not "
            "call this method and works today via MarkerProcessor. Implement "
            "gait event detection (heel-strike/toe-off per stride) here once "
            "the framework/insole-calibration integration is designed."
        )


class SkaterHop(Activity):
    """Lateral bounding: athlete lands on ALTERNATING legs (R, L, R, ...)
    within one trial -- same "both feet in one trial" shape as Walk, not a
    single committed side like SingleLegJump.

    Force-plate assignment is fixed per the study protocol: right foot on
    plates 3/1, left foot on plates 6/5, throughout the trial (whichever
    plate is loaded at a given instant tells you which foot is down).
    ForcePlateProcessor._resultant()'s existing multi-plate weighted-
    average combining handles this with no changes, exactly like Walk.

    Raw SKH filenames have no _L_/_R_ side segment (confirmed against real
    data), consistent with the alternating-leg shape -- trial.side is None
    and marker_plates() does not consult it. (An earlier stale assumption
    in subject.py's side_from_name() treated SKH as having one
    representative side via the subject's exo_leg metadata; that's been
    removed as factually wrong for an alternating-leg movement.)

    Framework/insole-calibration integration is NOT implemented here
    (synchronize() raises) -- unlike Walk, real OpenCap markerless
    coverage already exists for at least AB08's SKH sessions (see
    subject_rules/AB08.json's opencap_aliases), so this is near-term
    future work, not speculative. Whenever it's built: FrameworkProcessor
    currently reads a single trial.side unconditionally in a few places
    (picking which side's force/cop/contact to keep) -- that logic needs
    per-landing side detection across the trial's alternating stance
    events, not a single trial-wide side, or it will silently misattribute
    every landing to one foot.
    """

    name = "skater_hop"
    title = "Skater hop (lateral bounding, alternating legs)"
    token = "SKH"
    has_side_token = False
    unilateral_framework = False
    framework_interval = "not yet defined -- needs per-landing side detection, not a single trial.side"
    functional_outcomes = ()

    def marker_plates(self, trial):
        return ([3, 1], [6, 5])

    def synchronize(self, trial, trc, ik_all, insole_all, rules):
        raise NotImplementedError(
            "SkaterHop.synchronize() is not implemented -- marker-based "
            "processing (c3d convert -> force-plate combine -> scale/IK/ID) "
            "does not call this method and works today via MarkerProcessor. "
            "Framework integration needs per-landing (not per-trial) side "
            "detection across the alternating R/L stance events before this "
            "can be implemented -- see the class docstring."
        )


class LateralHop(Activity):
    """Single-leg side-to-side hopping: one committed side per trial (like
    SingleLegJump), but BOTH plates 1 and 6 belong to that SAME side
    throughout the hopping phase -- not a right/left split like SKH/Walk.

    Some trials have an initial and/or final zone where both feet are
    briefly on the plate together (a static double-support ready/landing
    hold) before/after the real single-leg hopping. contact_masks() (see
    Activity's base implementation) excludes those samples from the
    resultant by CONTACT DURATION: a genuine hop stance phase is a short,
    transient pulse (typically ~0.15-0.4s); a held double-support zone is
    much longer. MAX_SINGLE_SUPPORT_DURATION_S is an initial, tunable
    threshold -- verify it against real LH plate-contact durations rather
    than trusting it blindly (see this session's verification notes).

    Framework/insole-calibration integration is NOT implemented here
    (synchronize() raises) -- AB08/NOEXO already has real OpenCap coverage
    for lateral_hop too (subject_rules-adjacent opencap_sessions.csv), so
    this is near-term future work, not speculative.
    """

    name = "lateral_hop"
    title = "Lateral hop (single-leg side-to-side)"
    token = "LH"
    unilateral_framework = True
    framework_interval = "not yet defined -- framework/insole-calibration integration not implemented"
    functional_outcomes = ()

    # Verified against real data across AB03/AB05/AB08/AB10's LH trials:
    # genuine single-leg hop contacts range ~0.25-1.3s (varies by subject's
    # hop tempo); double-support holds (trial-start ready stance, and
    # sometimes a mid-trial reset) are all >=2.26s, up to 7.17s for AB10 --
    # a clean, consistent gap across every subject checked. 1.8s sits
    # comfortably in that gap. (An initial guess of 0.7s was WRONG -- it
    # discarded most genuine hop contacts, which commonly exceed 0.7s --
    # caught by inspecting real contact-duration data before trusting it.)
    MAX_SINGLE_SUPPORT_DURATION_S = 1.8

    def marker_plates(self, trial):
        return ([1, 6], []) if trial.side == "R" else ([], [1, 6])

    def contact_masks(self, trial, plate_forces: dict, dt: float):
        masks = {}
        for plate, force in plate_forces.items():
            active = np.abs(force) >= self.config.contact_force_n
            mask = np.zeros(len(force), dtype=bool)
            for start, end in io_tables.segments(active):
                duration = (end - start + 1) * dt
                if duration <= self.MAX_SINGLE_SUPPORT_DURATION_S:
                    mask[start:end + 1] = True
            masks[plate] = mask
        return masks

    def synchronize(self, trial, trc, ik_all, insole_all, rules):
        raise NotImplementedError(
            "LateralHop.synchronize() is not implemented -- marker-based "
            "processing (c3d convert -> force-plate combine -> scale/IK/ID) "
            "does not call this method and works today via MarkerProcessor. "
            "Framework/insole-calibration integration is deferred future work."
        )


_ACTIVITY_CLASSES = {
    "single_leg_jump": SingleLegJump,
    "slj": SingleLegJump,
    "walk": Walk,
    "walking": Walk,
    "skater_hop": SkaterHop,
    "skh": SkaterHop,
    "lateral_hop": LateralHop,
    "lh": LateralHop,
}


def activity_for(name: str, config=DEFAULT) -> Activity:
    key = name.lower()
    if key not in _ACTIVITY_CLASSES:
        raise ValueError(f"Unknown activity: {name}")
    return _ACTIVITY_CLASSES[key](config)
