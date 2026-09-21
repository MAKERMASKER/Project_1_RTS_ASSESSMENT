"""select_stance_windows.py -- manual, human-in-the-loop insole vertical-force
calibration window selection.

The automatic quality-gated stance detector in calibration.py
(_stance_window: opposite-side-near-zero + weight-bearing + low-gradient,
gated by run duration/CV/body-weight plausibility) still comes up empty for
several subjects/sides -- some trials genuinely have no long, flat,
plausible single-leg-stance window at all, and no amount of tuning finds
data that isn't there. This script lets a human do what the eye is still
better at than the heuristic: look at each trial's raw synced insole-vs-
force-plate plot (same style as collect_insole_raw_visualization.py) and
either mark a genuinely usable flat window with two clicks, or skip the
trial entirely if nothing in it is good enough -- exactly the selection
the user described from inspecting AB05_EXO_SLJ_R_4 (usable) vs.
AB05_EXO_SLJ_R_1 (not).

Run per subject (NTS01 excluded -- not part of this manual pass):
    python select_stance_windows.py --subject AB08

Controls:
    Click twice on the top panel to mark a window's start/end time --> a
    shaded span appears with live median/CV/%bodyweight feedback.
    [Save Window & Next]  keep this window for this trial, advance.
    [Skip Trial]           no usable window here, advance without saving.
    [Reset Selection]      clear the two clicks on the current trial.
    [Previous]             go back one trial (to redo a misclick).
    Closing the window early saves whatever was already decided.

Output (this folder):
    <SUBJECT>_manual_stance_selections.csv   one row per decided trial
        (Trial, Side, Selected, StartS, EndS, DurationS, MedianN, CV,
        WeightFraction)
    <SUBJECT>_manual_calibration.json        per-side gain computed from the
        accepted windows only, same formula as InsoleCalibration's auto
        fit (body_weight / median of accepted trials' medians) -- but never
        written into results/calibration/ or subject_rules by this script;
        promoting a manual gain into production is a separate, deliberate
        step.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from activities import activity_for  # noqa: E402
from config import DEFAULT  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from io_tables import read_insole, read_storage  # noqa: E402
from opensim_env import lowpass  # noqa: E402
from subject import Subject  # noqa: E402

ACTIVITY_NAME = "single_leg_jump"
OUT_DIR = Path(__file__).resolve().parent
LOWPASS_HZ = 6.0

COLOR_PLATE_R = "purple"
COLOR_PLATE_L = "red"
COLOR_INSOLE_R = "deepskyblue"
COLOR_INSOLE_L = "orange"


def _select_trials(subject, activity, discovery):
    trials = [
        trial
        for trial in discovery.trials(subject, activity, "all", for_analysis=True)
        if trial.insole.is_file() and trial.force_mot.is_file()
    ]
    return sorted(trials, key=lambda t: t.name)


def _shift_for(trial, insole, force_plate):
    """Same peak-alignment convention used throughout this project (see
    collect_insole_raw_visualization.py / calibration.py's _trial_delta)."""
    column = "left_force" if trial.side == "L" else "right_force"
    plate = 1 if trial.side == "R" else 2
    own_force = insole[column].to_numpy(float)
    own_plate = np.abs(force_plate[f"ground_force_{plate}_vy"].to_numpy(float))
    smoothed = np.convolve(own_force, np.ones(7) / 7, mode="same")
    return float(
        force_plate["time"].iloc[int(np.argmax(own_plate))]
        - insole["time"].iloc[int(np.argmax(smoothed))]
    )


def _trial_series(trial):
    insole = read_insole(trial.insole)
    force_plate = read_storage(trial.force_mot)
    shift = _shift_for(trial, insole, force_plate)

    time = insole["time"].to_numpy(float)
    insole_r = insole["right_force"].to_numpy(float)
    insole_l = insole["left_force"].to_numpy(float)
    plate_r_raw = np.abs(force_plate["ground_force_1_vy"].to_numpy(float))
    plate_l_raw = np.abs(force_plate["ground_force_2_vy"].to_numpy(float))
    plate_r = np.interp(time + shift, force_plate["time"], plate_r_raw, left=np.nan, right=np.nan)
    plate_l = np.interp(time + shift, force_plate["time"], plate_l_raw, left=np.nan, right=np.nan)
    own = insole_r if trial.side == "R" else insole_l
    return {
        "time": time, "insole_r": insole_r, "insole_l": insole_l,
        "plate_r": plate_r, "plate_l": plate_l, "own": own,
    }


class StanceSelector:
    def __init__(self, subject_id):
        self.subject_id = subject_id
        self.subject = Subject(subject_id, DEFAULT)
        self.weight_n = self.subject.mass_kg * DEFAULT.gravity_mps2 if self.subject.mass_kg else None
        activity = activity_for(ACTIVITY_NAME, DEFAULT)
        discovery = TrialDiscovery(DEFAULT)
        self.trials = _select_trials(self.subject, activity, discovery)
        if not self.trials:
            raise SystemExit(f"{subject_id}: no eligible trials (missing insole/force_mot)")

        self.index = 0
        self.clicks = []
        self.decisions = {}  # trial name -> record dict

        self.figure, (self.top, self.bottom) = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True)
        plt.subplots_adjust(bottom=0.22)
        self.span = None
        self.feedback_text = None

        button_specs = [
            ("Previous", self._on_previous),
            ("Reset Selection", self._on_reset),
            ("Skip Trial", self._on_skip),
            ("Save Window && Next", self._on_save),
        ]
        self.buttons = []
        left = 0.15
        width = 0.18
        for label, handler in button_specs:
            axis = self.figure.add_axes([left, 0.05, width, 0.06])
            button = Button(axis, label)
            button.on_clicked(handler)
            self.buttons.append(button)
            left += width + 0.02

        self.figure.canvas.mpl_connect("button_press_event", self._on_click)
        self.figure.canvas.mpl_connect("close_event", self._on_close)
        self._draw_trial()

    # ---- data/drawing --------------------------------------------------

    @staticmethod
    def _safe_remove(artist):
        """artist.remove() raises NotImplementedError if it was already
        detached (e.g. by a prior axes.clear()) -- swallow that, this is
        always called to make sure something is gone, not to prove it."""
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
        self.top.clear()
        self.bottom.clear()

        trial = self.trials[self.index]
        series = _trial_series(trial)
        self._series = series
        self._trial = trial

        self.top.plot(series["time"], series["plate_r"], color=COLOR_PLATE_R, label="Force plate R", linewidth=1.2)
        self.top.plot(series["time"], series["plate_l"], color=COLOR_PLATE_L, label="Force plate L", linewidth=1.2)
        self.top.plot(series["time"], series["insole_r"], color=COLOR_INSOLE_R, label="Insole R (raw)", linewidth=1)
        self.top.plot(series["time"], series["insole_l"], color=COLOR_INSOLE_L, label="Insole L (raw)", linewidth=1)
        self.top.set_ylabel("Force (N)")
        decided = self.decisions.get(trial.name)
        status = ""
        if decided is not None:
            status = " -- SAVED" if decided["Selected"] else " -- SKIPPED"
        self.top.set_title(
            f"[{self.index + 1}/{len(self.trials)}] {trial.name} (side {trial.side}){status}"
            "  -- click twice to mark a window"
        )
        self.top.grid(True, alpha=0.3)
        self.top.legend(fontsize=8, ncol=2, loc="upper right")

        total_plate = series["plate_r"] + series["plate_l"]
        total_insole = series["insole_r"] + series["insole_l"]
        self.bottom.plot(series["time"], total_plate, color=COLOR_PLATE_R, label="Force plate TOTAL", linewidth=1.2)
        self.bottom.plot(series["time"], total_insole, color=COLOR_INSOLE_R, label="Insole TOTAL (raw)", linewidth=1.2)
        self.bottom.set_xlabel("Time (s)")
        self.bottom.set_ylabel("Force (N)")
        self.bottom.grid(True, alpha=0.3)
        self.bottom.legend(fontsize=8)

        if decided is not None and decided["Selected"]:
            self.clicks = [decided["StartS"], decided["EndS"]]
            self._draw_span_and_feedback()

        self.figure.canvas.draw_idle()

    def _draw_span_and_feedback(self):
        self._safe_remove(self.span)
        self.span = None
        start, end = sorted(self.clicks)
        self.span = self.top.axvspan(start, end, color="green", alpha=0.15)
        stats = self._window_stats(start, end)
        self._safe_remove(self.feedback_text)
        self.feedback_text = self.top.text(
            0.01, 0.02,
            f"window {stats['duration']:.2f}s  median={stats['median']:.0f}N  "
            f"CV={stats['cv']:.1%}  {stats['weight_fraction']:.0%} BW",
            transform=self.top.transAxes, fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="green"),
        )
        self.figure.canvas.draw_idle()

    def _window_stats(self, start, end):
        time = self._series["time"]
        own = self._series["own"]
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
        if event.inaxes is not self.top or len(self.clicks) >= 2:
            return
        if event.xdata is None:
            return
        self.clicks.append(float(event.xdata))
        if len(self.clicks) == 2:
            self._draw_span_and_feedback()
        self.figure.canvas.draw_idle()

    def _on_reset(self, _event):
        self.clicks = []
        self._safe_remove(self.span)
        self.span = None
        self._safe_remove(self.feedback_text)
        self.feedback_text = None
        self.figure.canvas.draw_idle()

    def _on_skip(self, _event):
        trial = self._trial
        self.decisions[trial.name] = {
            "Trial": trial.name, "Side": trial.side, "Selected": False,
            "StartS": np.nan, "EndS": np.nan, "DurationS": np.nan,
            "MedianN": np.nan, "MedianFilteredN": np.nan, "CV": np.nan, "WeightFraction": np.nan,
        }
        self._advance()

    def _on_save(self, _event):
        if len(self.clicks) != 2:
            print("  (need two clicks to define a window before saving -- ignored)")
            return
        trial = self._trial
        start, end = sorted(self.clicks)
        stats = self._window_stats(start, end)
        self.decisions[trial.name] = {
            "Trial": trial.name, "Side": trial.side, "Selected": True,
            "StartS": start, "EndS": end, "DurationS": stats["duration"],
            "MedianN": stats["median"], "MedianFilteredN": stats["median_filtered"],
            "CV": stats["cv"], "WeightFraction": stats["weight_fraction"],
        }
        self._advance()

    def _on_previous(self, _event):
        if self.index > 0:
            self.index -= 1
            self._draw_trial()

    def _advance(self):
        if self.index + 1 < len(self.trials):
            self.index += 1
            self._draw_trial()
        else:
            print(f"{self.subject_id}: reached the last trial -- close the window to save results.")

    def _on_close(self, _event):
        self._save_results()

    # ---- output ------------------------------------------------------

    def _save_results(self):
        import pandas as pd

        if not self.decisions:
            print(f"{self.subject_id}: no trials were decided, nothing saved.")
            return

        rows = [self.decisions[t.name] for t in self.trials if t.name in self.decisions]
        table = pd.DataFrame(rows)
        selections_path = OUT_DIR / f"{self.subject_id}_manual_stance_selections.csv"
        table.to_csv(selections_path, index=False)

        calibration = {}
        for side in ("L", "R"):
            accepted = table[(table["Side"] == side) & (table["Selected"])]
            if len(accepted) == 0 or self.weight_n is None:
                calibration[side] = {"accepted_trials": 0, "gain": None, "note": "no manually accepted window"}
                continue
            median = float(np.median(accepted["MedianFilteredN"]))
            calibration[side] = {
                "accepted_trials": int(len(accepted)),
                "trials": accepted["Trial"].tolist(),
                "median_of_accepted_n": median,
                "body_weight_n": self.weight_n,
                "gain": self.weight_n / median,
                "source": "manual_stance_selection",
            }
        calibration_path = OUT_DIR / f"{self.subject_id}_manual_calibration.json"
        calibration_path.write_text(json.dumps(calibration, indent=2))

        print(f"{self.subject_id}: {len(rows)} trials decided -> {selections_path}")
        print(f"{self.subject_id}: calibration -> {calibration_path}")
        for side, info in calibration.items():
            if info.get("gain") is not None:
                print(f"  {side}: gain={info['gain']:.4f} from {info['accepted_trials']} trial(s) {info['trials']}")
            else:
                print(f"  {side}: no usable manual window")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, help="e.g. AB08 (NTS01 excluded from this manual pass)")
    args = parser.parse_args()
    if args.subject.upper() == "NTS01":
        raise SystemExit("NTS01 is excluded from manual stance-window selection.")
    selector = StanceSelector(args.subject)
    plt.show()


if __name__ == "__main__":
    main()
