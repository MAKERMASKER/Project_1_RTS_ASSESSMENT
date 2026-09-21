"""select_walk_stance_windows.py -- manual, human-in-the-loop insole
vertical-force calibration window selection FROM WALKING TRIALS.

Companion to select_stance_windows.py (the SLJ-based version). Scope: only
the subjects whose insole gain still carries real residual uncertainty
after the SLJ-based manual calibration --
AB03 (-11 to -20% residual peak-GRF error), AB05 (18-37% residual either
way), and AB10's L side specifically (every gain tried made it WORSE than
raw). AB07 and AB08 were already resolved cleanly from SLJ trials (see
select_stance_windows.py's outputs and each subject_rules/*.json's
force_scale_note) and don't need this. AB10's R side is also already
resolved (kept via subject_rules, not touched here).

Restricted further to the walking trials that survived sync validation
this session (see collect_walk_insole_raw_visualization.py's
OMITTED_TRIALS and the manual review that followed): AB03 WALK_1+WALK_2
(solid), AB05 WALK_2 only (WALK_1's insole recording never overlaps the
plate-active window), AB10 WALK_2 only (WALK_1 gave an ambiguous,
trial-inconsistent plate<->insole correlation and was omitted).

Sync reuses collect_walk_insole_raw_visualization.py's
_cross_correlation_shift() (constrained to a small window around the
"insole and mocap started together" prior) and _trial_series() verbatim --
not reimplemented, so this is guaranteed consistent with what was already
visually reviewed and approved.

Differences from the SLJ selector:
  - A walking trial has MULTIPLE alternating L/R stance phases, not one --
    trial.side is None for Walk (see activities.py's Walk class), so this
    lets you save any number of windows per trial, each explicitly tagged
    Left or Right, instead of one window per trial on a side fixed by the
    trial's own name.
  - Output is explicitly tagged with its data source (source_activity=
    "walk") so it's never confused with the SLJ-derived calibration when
    both exist for the same subject.

Run per subject:
    python select_walk_stance_windows.py --subject AB03

Controls:
    Click twice on the panel to mark a window --> a shaded span appears
    with live median/CV/%bodyweight feedback.
    [Save as LEFT] / [Save as RIGHT]   keep this window, tagged to that
        side; clicks reset so you can immediately mark another window in
        the same trial (a walking trial has several stance phases).
    [Reset Selection]                  clear the two pending clicks.
    [Undo Last Window]                 remove the most recently saved
        window in this trial.
    [Previous Trial] / [Next Trial]    navigate explicitly (no forced
        auto-advance after a save, unlike the SLJ selector).
    Closing the window saves everything decided so far.

Output (this folder):
    <SUBJECT>_walk_manual_stance_selections.csv   one row per saved window
        (Trial, Side, StartS, EndS, DurationS, MedianN, MedianFilteredN,
        CV, WeightFraction)
    <SUBJECT>_walk_manual_calibration.json        per-side gain computed
        from ALL saved windows for that side, pooled across trials --
        same formula as the SLJ selector (body_weight / median of the
        windows' own medians) -- explicitly tagged source_activity="walk"
        and source_script=this file's path, so its provenance is always
        traceable. Never written into results/calibration/ or
        subject_rules by this script; promoting a gain into production is
        a separate, deliberate step.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from activities import activity_for  # noqa: E402
from config import DEFAULT  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from io_tables import read_insole, read_storage  # noqa: E402
from opensim_env import lowpass  # noqa: E402
from subject import Subject  # noqa: E402

# Deliberately NOT imported from collect_walk_insole_raw_visualization.py:
# that module calls matplotlib.use("Agg") at import time (needed for its
# own headless PNG generation), which would silently disable plt.show()
# here and leave the GUI window never appearing -- same reason
# select_stance_windows.py duplicates its sync logic instead of importing
# collect_insole_raw_visualization.py. The functions below are copied
# verbatim from collect_walk_insole_raw_visualization.py (same file this
# session already validated the sync against) -- keep them in sync if that
# file's algorithm changes.

ACTIVITY_NAME = "walk"
OUT_DIR = Path(__file__).resolve().parent
LOWPASS_HZ = 6.0
SOURCE_SCRIPT = "code/expt_temp/Insole_vertical_force_calib_window/select_walk_stance_windows.py"
SYNC_PRIOR_MARGIN_S = 0.75

COLOR_PLATE_R = "purple"
COLOR_PLATE_L = "red"
COLOR_INSOLE_R = "deepskyblue"
COLOR_INSOLE_L = "orange"
SPAN_COLOR = {"L": "orange", "R": "deepskyblue"}

# Same omissions as collect_walk_insole_raw_visualization.py -- see that
# module's OMITTED_TRIALS docstring for the full rationale (AB05_WALK_1:
# insole recording doesn't overlap the plate-active window at all;
# AB10_WALK_1: ambiguous/trial-inconsistent sync, omitted per the user).
OMITTED_TRIALS = {"AB05_NOEXO_WALK_1", "AB10_NOEXO_WALK_1"}


def _select_trials(subject, activity, discovery):
    trials = [
        trial
        for trial in discovery.trials(subject, activity, "NOEXO", for_analysis=True)
        if trial.insole.is_file() and trial.force_mot.is_file() and trial.name not in OMITTED_TRIALS
    ]
    return sorted(trials, key=lambda t: t.name)


def _cross_correlation_shift(insole_time, insole_total, plate_time, plate_total):
    """Time to ADD to insole_time so it aligns with plate_time -- cross-
    correlation constrained to SYNC_PRIOR_MARGIN_S around the "insole and
    mocap started together" prior (see collect_walk_insole_raw_visualization
    .py's docstring for the full rationale: gait is periodic, so an
    unconstrained search can lock onto the wrong stride)."""
    dt = float(np.median(np.diff(insole_time)))
    plate_uniform_time = np.arange(plate_time.min(), plate_time.max(), dt)
    plate_uniform = np.interp(plate_uniform_time, plate_time, plate_total)

    insole_centered = insole_total - np.mean(insole_total)
    plate_centered = plate_uniform - np.mean(plate_uniform)

    correlation = np.correlate(plate_centered, insole_centered, mode="full")
    center_index = len(insole_centered) - 1
    margin_samples = int(round(SYNC_PRIOR_MARGIN_S / dt))
    low = max(0, center_index - margin_samples)
    high = min(len(correlation) - 1, center_index + margin_samples)
    best_index = low + int(np.argmax(correlation[low:high + 1]))
    lag_samples = best_index - center_index

    prior_shift = plate_uniform_time[0] - insole_time[0]
    return prior_shift + lag_samples * dt


def _trial_series(trial):
    insole = read_insole(trial.insole)
    force_plate = read_storage(trial.force_mot)

    insole_total = insole["right_force"].to_numpy(float) + insole["left_force"].to_numpy(float)
    plate_r_raw = np.abs(force_plate["ground_force_1_vy"].to_numpy(float))
    plate_l_raw = np.abs(force_plate["ground_force_2_vy"].to_numpy(float))
    plate_total = plate_r_raw + plate_l_raw

    shift = _cross_correlation_shift(
        insole["time"].to_numpy(float), insole_total,
        force_plate["time"].to_numpy(float), plate_total,
    )

    time = insole["time"].to_numpy(float)
    insole_r = insole["right_force"].to_numpy(float)
    insole_l = insole["left_force"].to_numpy(float)
    plate_r = np.interp(time + shift, force_plate["time"], plate_r_raw, left=np.nan, right=np.nan)
    plate_l = np.interp(time + shift, force_plate["time"], plate_l_raw, left=np.nan, right=np.nan)
    return {
        "time": time, "insole_r": insole_r, "insole_l": insole_l,
        "plate_r": plate_r, "plate_l": plate_l, "shift": shift,
    }


class WalkStanceSelector:
    def __init__(self, subject_id):
        self.subject_id = subject_id
        self.subject = Subject(subject_id, DEFAULT)
        self.weight_n = self.subject.mass_kg * DEFAULT.gravity_mps2 if self.subject.mass_kg else None
        activity = activity_for(ACTIVITY_NAME, DEFAULT)
        discovery = TrialDiscovery(DEFAULT)
        self.trials = _select_trials(self.subject, activity, discovery)
        if not self.trials:
            raise SystemExit(f"{subject_id}: no eligible WALK trials (missing insole/force_mot, or all omitted)")
        print(f"{subject_id}: {len(self.trials)} trial(s) -> {[t.name for t in self.trials]}")

        self.index = 0
        self.clicks = []
        # trial name -> list of saved window dicts (unlike the SLJ selector,
        # a walking trial can contribute any number of windows)
        self.decisions = {}

        self.figure, self.axis = plt.subplots(1, 1, figsize=(13, 7.5))
        plt.subplots_adjust(bottom=0.28)
        self.span = None
        self.feedback_text = None
        self.saved_spans = []

        button_specs = [
            ("Previous Trial", self._on_previous),
            ("Reset Selection", self._on_reset),
            ("Undo Last Window", self._on_undo),
            ("Save as LEFT", lambda event: self._on_save(event, "L")),
            ("Save as RIGHT", lambda event: self._on_save(event, "R")),
            ("Next Trial", self._on_next),
        ]
        self.buttons = []
        left = 0.06
        width = 0.145
        for label, handler in button_specs:
            axis = self.figure.add_axes([left, 0.08, width, 0.07])
            button = Button(axis, label)
            button.on_clicked(handler)
            self.buttons.append(button)
            left += width + 0.015

        self.figure.canvas.mpl_connect("button_press_event", self._on_click)
        self.figure.canvas.mpl_connect("close_event", self._on_close)
        self._draw_trial()

    # ---- data/drawing --------------------------------------------------

    @staticmethod
    def _safe_remove(artist):
        if artist is None:
            return
        try:
            artist.remove()
        except NotImplementedError:
            pass

    def _draw_trial(self):
        self.clicks = []
        self.span = None
        self.feedback_text = None
        self.saved_spans = []
        self.axis.clear()

        trial = self.trials[self.index]
        series = _trial_series(trial)
        self._series = series
        self._trial = trial

        self.axis.plot(series["time"], series["plate_r"], color=COLOR_PLATE_R, label="Force plate R", linewidth=1.2)
        self.axis.plot(series["time"], series["plate_l"], color=COLOR_PLATE_L, label="Force plate L", linewidth=1.2)
        self.axis.plot(series["time"], series["insole_r"], color=COLOR_INSOLE_R, label="Insole R (raw)", linewidth=1)
        self.axis.plot(series["time"], series["insole_l"], color=COLOR_INSOLE_L, label="Insole L (raw)", linewidth=1)
        self.axis.set_ylabel("Force (N)")
        self.axis.set_xlabel("Time (s)")
        saved = self.decisions.get(trial.name, [])
        self.axis.set_title(
            f"[{self.index + 1}/{len(self.trials)}] {trial.name} -- {len(saved)} window(s) saved"
            "  -- click twice to mark a window, then Save as LEFT/RIGHT"
        )
        self.axis.grid(True, alpha=0.3)
        self.axis.legend(fontsize=8, ncol=2, loc="upper right")

        for record in saved:
            self._draw_saved_span(record)

        self.figure.canvas.draw_idle()

    def _draw_saved_span(self, record):
        span = self.axis.axvspan(record["StartS"], record["EndS"], color=SPAN_COLOR[record["Side"]], alpha=0.2)
        self.saved_spans.append(span)
        mid = (record["StartS"] + record["EndS"]) / 2
        self.axis.annotate(
            record["Side"], (mid, self.axis.get_ylim()[1] * 0.95),
            ha="center", fontsize=9, fontweight="bold",
        )

    def _draw_span_and_feedback(self, side):
        self._safe_remove(self.span)
        self.span = None
        start, end = sorted(self.clicks)
        self.span = self.axis.axvspan(start, end, color="green", alpha=0.15)
        stats = self._window_stats(start, end, side)
        self._safe_remove(self.feedback_text)
        self.feedback_text = self.axis.text(
            0.01, 0.02,
            f"[{side}] window {stats['duration']:.2f}s  median={stats['median']:.0f}N  "
            f"CV={stats['cv']:.1%}  {stats['weight_fraction']:.0%} BW",
            transform=self.axis.transAxes, fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="green"),
        )
        self.figure.canvas.draw_idle()

    def _window_stats(self, start, end, side):
        time = self._series["time"]
        own = self._series["insole_r"] if side == "R" else self._series["insole_l"]
        mask = (time >= start) & (time <= end) & np.isfinite(own)
        segment_raw = own[mask]
        filtered = lowpass(own, time, LOWPASS_HZ)
        segment = filtered[mask]
        mean = float(np.mean(segment)) if len(segment) else float("nan")
        cv = float(np.std(segment) / mean) if segment.size and mean else float("nan")
        weight_fraction = mean / self.weight_n if self.weight_n else float("nan")
        return {
            "duration": end - start, "median": float(np.median(segment_raw)) if len(segment_raw) else float("nan"),
            "median_filtered": float(np.median(segment)) if len(segment) else float("nan"),
            "cv": cv, "weight_fraction": weight_fraction,
        }

    # ---- interaction -----------------------------------------------------

    def _on_click(self, event):
        if event.inaxes is not self.axis or len(self.clicks) >= 2:
            return
        if event.xdata is None:
            return
        self.clicks.append(float(event.xdata))
        if len(self.clicks) == 2:
            # No side chosen yet -- draw a neutral pending span; feedback
            # text will be redrawn with real per-side stats on Save.
            self._draw_span_and_feedback("?")
        self.figure.canvas.draw_idle()

    def _on_reset(self, _event):
        self.clicks = []
        self._safe_remove(self.span)
        self.span = None
        self._safe_remove(self.feedback_text)
        self.feedback_text = None
        self.figure.canvas.draw_idle()

    def _on_undo(self, _event):
        trial = self._trial
        records = self.decisions.get(trial.name, [])
        if records:
            records.pop()
        self._draw_trial()

    def _on_save(self, _event, side):
        if len(self.clicks) != 2:
            print("  (need two clicks to define a window before saving -- ignored)")
            return
        trial = self._trial
        start, end = sorted(self.clicks)
        stats = self._window_stats(start, end, side)
        record = {
            "Trial": trial.name, "Side": side,
            "StartS": start, "EndS": end, "DurationS": stats["duration"],
            "MedianN": stats["median"], "MedianFilteredN": stats["median_filtered"],
            "CV": stats["cv"], "WeightFraction": stats["weight_fraction"],
        }
        self.decisions.setdefault(trial.name, []).append(record)
        print(f"  {trial.name}: saved {side} window {start:.2f}-{end:.2f}s "
              f"(median={stats['median']:.0f}N, {stats['weight_fraction']:.0%} BW)")
        self.clicks = []
        self._draw_trial()

    def _on_previous(self, _event):
        if self.index > 0:
            self.index -= 1
            self._draw_trial()

    def _on_next(self, _event):
        if self.index + 1 < len(self.trials):
            self.index += 1
            self._draw_trial()
        else:
            print(f"{self.subject_id}: reached the last trial -- close the window to save results.")

    def _on_close(self, _event):
        self._save_results()

    # ---- output ------------------------------------------------------

    def _save_results(self):
        rows = [record for records in self.decisions.values() for record in records]
        if not rows:
            print(f"{self.subject_id}: no windows were saved, nothing to write.")
            return

        table = pd.DataFrame(rows)
        selections_path = OUT_DIR / f"{self.subject_id}_walk_manual_stance_selections.csv"
        table.to_csv(selections_path, index=False)

        calibration = {
            "subject": self.subject_id,
            "source_activity": "walk",
            "source_script": SOURCE_SCRIPT,
            "note": (
                "Gain fit from manually-selected quiet single-limb-stance windows "
                "within WALKING trials (natural gait stance phase), not SLJ jump "
                "trials -- see this subject's SLJ-derived calibration (if any) in "
                "the sibling select_stance_windows.py output / subject_rules "
                "force_scale_note for comparison. Formula identical either way: "
                "gain = body_weight_n / median(saved windows' own median force)."
            ),
        }
        for side in ("L", "R"):
            accepted = table[table["Side"] == side]
            if len(accepted) == 0 or self.weight_n is None:
                calibration[side] = {"accepted_windows": 0, "gain": None, "note": "no manually saved window"}
                continue
            median = float(np.median(accepted["MedianFilteredN"]))
            calibration[side] = {
                "accepted_windows": int(len(accepted)),
                "trials": sorted(accepted["Trial"].unique().tolist()),
                "median_of_accepted_n": median,
                "body_weight_n": self.weight_n,
                "gain": self.weight_n / median,
                "source": "manual_stance_selection",
            }
        calibration_path = OUT_DIR / f"{self.subject_id}_walk_manual_calibration.json"
        calibration_path.write_text(json.dumps(calibration, indent=2))

        print(f"{self.subject_id}: {len(rows)} windows saved -> {selections_path}")
        print(f"{self.subject_id}: calibration -> {calibration_path}")
        for side in ("L", "R"):
            info = calibration[side]
            if info.get("gain") is not None:
                print(f"  {side}: gain={info['gain']:.4f} from {info['accepted_windows']} window(s), trials {info['trials']}")
            else:
                print(f"  {side}: no usable manual window")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, help="AB03, AB05, or AB10 -- the only subjects in scope")
    args = parser.parse_args()
    WalkStanceSelector(args.subject)
    plt.show()


if __name__ == "__main__":
    main()
