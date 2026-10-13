"""select_skh_sync.py -- manual, human-in-the-loop insole<->OpenCap time
synchronization for Skater Hop (SKH) trials.

Why this exists: SkaterHop.synchronize()'s automatic liftoff/landing
two-anchor detection (see activities.py) was initially judged reliable, but
once run across all 6 subjects only 6/22 trials (27%) agreed within the
0.2s cross-check tolerance (16/22 flagged quality_omit=True) -- the same
"good in principle, unreliable in practice on this data" pattern already
found for LateralHop. Same fix: sync manually by eye, same tool design as
select_lh_sync.py (see that script's own docstring for the full LH
history), just retargeted at skater_hop trials.

Design, identical to select_lh_sync.py:
  - Show the first WINDOW_SECONDS (15s) of EACH signal's own NATIVE/local
    time (not an already-shifted or absolute clock).
  - Two panels: OpenCap (top: right/left foot height) and Insole (bottom:
    right/left raw force). Click ONCE on each panel to mark the same
    real-world event in both domains -- for SKH, the clearest shared event
    is normally the right foot's first landing (per the study protocol:
    right foot lifts off, left foot initiates the hop, right foot lands
    first), but click whichever event is clearest in both traces.
  - [Save Sync & Next] records shift = opencap_click_time -
    insole_click_time (same convention as activities.py's _finish_sync).
  - [Reject Trial & Next] marks a trial unusable without recording a
    shift -- intended to be promoted into subject_rules/<ID>.json's
    "exclude" list once reviewed (nothing here writes to subject_rules
    directly).

Run per subject (from code/, in the opensimPy environment):
    python expt_temp/skh_manual_sync/select_skh_sync.py --subject AB08

Controls:
    Click once on the TOP (OpenCap) panel, once on the BOTTOM (Insole)
    panel -- a vertical line marks each click.
    [Reset Clicks]          clear both pending clicks for this trial.
    [Save Sync && Next]     record the shift, advance (needs both clicks).
    [Reject Trial && Next]  mark this trial unusable, advance (no clicks
        needed).
    [Previous Trial]        go back one trial to redo a misclick/decision.
    Closing the window saves everything decided so far.

Output (this folder):
    <SUBJECT>_skh_manual_sync.csv -- one row per decided trial: Trial,
        Status (synced/rejected), OpenCapClickTimeS, InsoleClickTimeS,
        ShiftS (OpenCapClickTimeS - InsoleClickTimeS, synced rows only).
"""
import argparse
import csv
import functools
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from activities import activity_for  # noqa: E402
from config import DEFAULT  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from insole_geometry import InsoleGeometry  # noqa: E402
from io_tables import read_insole, read_trc  # noqa: E402
from subject import Subject  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
ACTIVITY_NAME = "skater_hop"
WINDOW_SECONDS = 15.0

COLOR_R = "purple"
COLOR_L = "orange"


def _guarded(method):
    """Catch+report any exception in a GUI callback instead of letting it
    propagate silently -- an uncaught exception inside a matplotlib/Tkinter
    event callback can leave the event loop's internal state (e.g. button
    grab) stuck, which looks like a total freeze (no button/click response)
    rather than a visible crash."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception:
            print(f"[select_skh_sync] error in {method.__name__} -- ignored, GUI stays open:")
            traceback.print_exc()
            return None

    return wrapper


def _select_trials(subject, activity, discovery):
    trials = []
    for trial in discovery.trials(subject, activity, "all", for_analysis=True):
        opencap_trc = trial.effective_opencap_trc()
        if opencap_trc and Path(opencap_trc).is_file() and trial.insole.is_file():
            trials.append(trial)
    return sorted(trials, key=lambda t: t.name)


def _trial_series(trial):
    trc = read_trc(trial.effective_opencap_trc())
    insole_all = read_insole(trial.insole)

    opencap_time = trc["time"]
    opencap_mask = opencap_time <= opencap_time.min() + WINDOW_SECONDS
    insole_time = insole_all["time"].to_numpy(float)
    insole_mask = insole_time <= insole_time.min() + WINDOW_SECONDS

    return {
        "opencap_time": opencap_time[opencap_mask],
        "right_height": InsoleGeometry.foot_height(trc, "R")[opencap_mask],
        "left_height": InsoleGeometry.foot_height(trc, "L")[opencap_mask],
        "insole_time": insole_time[insole_mask],
        "right_force": insole_all["right_force"].to_numpy(float)[insole_mask],
        "left_force": insole_all["left_force"].to_numpy(float)[insole_mask],
    }


class SkhSyncSelector:
    def __init__(self, subject_id):
        self.subject_id = subject_id
        subject = Subject(subject_id, DEFAULT)
        activity = activity_for(ACTIVITY_NAME, DEFAULT)
        discovery = TrialDiscovery(DEFAULT)
        self.trials = _select_trials(subject, activity, discovery)
        if not self.trials:
            raise SystemExit(f"{subject_id}: no eligible SKH trials (missing OpenCap TRC/insole)")
        print(f"{subject_id}: {len(self.trials)} trial(s) -> {[t.name for t in self.trials]}")

        self.index = 0
        self.opencap_click = None
        self.insole_click = None
        self.decisions = {}  # trial name -> record dict

        self.figure, (self.top, self.bottom) = plt.subplots(2, 1, figsize=(12, 8))
        plt.subplots_adjust(bottom=0.25, hspace=0.55)
        self.opencap_line = None
        self.insole_line = None

        button_specs = [
            ("Previous Trial", self._on_previous),
            ("Reset Clicks", self._on_reset),
            ("Reject Trial && Next", self._on_reject),
            ("Save Sync && Next", self._on_save),
        ]
        self.buttons = []
        left = 0.12
        width = 0.19
        for label, handler in button_specs:
            axis = self.figure.add_axes([left, 0.06, width, 0.07])
            button = Button(axis, label)
            button.on_clicked(handler)
            self.buttons.append(button)
            left += width + 0.02

        self.figure.canvas.mpl_connect("button_press_event", self._on_click)
        self.figure.canvas.mpl_connect("close_event", self._on_close)
        self._draw_trial()

    # ---- drawing --------------------------------------------------

    @staticmethod
    def _safe_remove(artist):
        if artist is None:
            return
        try:
            artist.remove()
        except NotImplementedError:
            pass

    @_guarded
    def _draw_trial(self):
        self.opencap_click = None
        self.insole_click = None
        self.opencap_line = None
        self.insole_line = None
        self.top.clear()
        self.bottom.clear()

        trial = self.trials[self.index]
        series = _trial_series(trial)
        self._series = series
        self._trial = trial

        self.top.plot(series["opencap_time"], series["right_height"], color=COLOR_R, label="Right foot height")
        self.top.plot(series["opencap_time"], series["left_height"], color=COLOR_L, label="Left foot height")
        self.top.set_ylabel("Height (m)")
        self.top.set_xlabel("OpenCap time (s, own clock)")
        decided = self.decisions.get(trial.name)
        status = f" -- {decided['Status'].upper()}" if decided else ""
        self.top.set_title(
            f"[{self.index + 1}/{len(self.trials)}] {trial.name}{status} -- click ONE point (OpenCap event)"
        )
        self.top.grid(True, alpha=0.3)
        self.top.legend(fontsize=8, loc="upper right")

        self.bottom.plot(series["insole_time"], series["right_force"], color=COLOR_R, label="Right insole (raw)")
        self.bottom.plot(series["insole_time"], series["left_force"], color=COLOR_L, label="Left insole (raw)")
        self.bottom.set_ylabel("Force (N)")
        self.bottom.set_xlabel("Insole time (s, own clock)")
        self.bottom.set_title("click ONE point (the SAME real-world event, insole side)")
        self.bottom.grid(True, alpha=0.3)
        self.bottom.legend(fontsize=8, loc="upper right")

        if decided and decided["Status"] == "synced":
            self.opencap_click = decided["OpenCapClickTimeS"]
            self.insole_click = decided["InsoleClickTimeS"]
            self._draw_markers()

        self.figure.canvas.draw()

    def _draw_markers(self):
        self._safe_remove(self.opencap_line)
        self._safe_remove(self.insole_line)
        self.opencap_line = None
        self.insole_line = None
        if self.opencap_click is not None:
            self.opencap_line = self.top.axvline(self.opencap_click, color="green", linewidth=1.5)
        if self.insole_click is not None:
            self.insole_line = self.bottom.axvline(self.insole_click, color="green", linewidth=1.5)
        self.figure.canvas.draw()

    # ---- interaction -----------------------------------------------------

    @_guarded
    def _on_click(self, event):
        print(f"[select_skh_sync] click: xdata={event.xdata!r} inaxes={event.inaxes!r} "
              f"(top={self.top!r}, bottom={self.bottom!r})")
        if event.xdata is None:
            return
        if event.inaxes is self.top:
            self.opencap_click = float(event.xdata)
            self._draw_markers()
        elif event.inaxes is self.bottom:
            self.insole_click = float(event.xdata)
            self._draw_markers()

    @_guarded
    def _on_reset(self, _event):
        self.opencap_click = None
        self.insole_click = None
        self._safe_remove(self.opencap_line)
        self._safe_remove(self.insole_line)
        self.opencap_line = None
        self.insole_line = None
        self.figure.canvas.draw()

    @_guarded
    def _on_reject(self, _event):
        trial = self._trial
        self.decisions[trial.name] = {
            "Trial": trial.name, "Status": "rejected",
            "OpenCapClickTimeS": None, "InsoleClickTimeS": None, "ShiftS": None,
        }
        print(f"  {trial.name}: REJECTED")
        self._advance()

    @_guarded
    def _on_save(self, _event):
        if self.opencap_click is None or self.insole_click is None:
            print("  (need one click on each panel before saving -- ignored)")
            return
        trial = self._trial
        shift = self.opencap_click - self.insole_click
        self.decisions[trial.name] = {
            "Trial": trial.name, "Status": "synced",
            "OpenCapClickTimeS": self.opencap_click, "InsoleClickTimeS": self.insole_click,
            "ShiftS": shift,
        }
        print(f"  {trial.name}: synced, shift={shift:.3f}s "
              f"(opencap={self.opencap_click:.3f}s, insole={self.insole_click:.3f}s)")
        self._advance()

    @_guarded
    def _on_previous(self, _event):
        if self.index > 0:
            self.index -= 1
            self._draw_trial()

    @_guarded
    def _advance(self):
        if self.index + 1 < len(self.trials):
            self.index += 1
            self._draw_trial()
        else:
            print(f"{self.subject_id}: reached the last trial -- close the window to save results.")

    @_guarded
    def _on_close(self, _event):
        self._save_results()

    # ---- output ------------------------------------------------------

    def _save_results(self):
        if not self.decisions:
            print(f"{self.subject_id}: no trials were decided, nothing saved.")
            return

        rows = [self.decisions[t.name] for t in self.trials if t.name in self.decisions]
        path = OUT_DIR / f"{self.subject_id}_skh_manual_sync.csv"
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["Trial", "Status", "OpenCapClickTimeS", "InsoleClickTimeS", "ShiftS"])
            writer.writeheader()
            writer.writerows(rows)

        synced = [r for r in rows if r["Status"] == "synced"]
        rejected = [r for r in rows if r["Status"] == "rejected"]
        print(f"{self.subject_id}: {len(rows)} trials decided -> {path}")
        print(f"  synced: {len(synced)}, rejected: {len(rejected)}")
        if rejected:
            print(f"  rejected trials (candidates for subject_rules 'exclude'): "
                  f"{[r['Trial'] for r in rejected]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, help="e.g. AB08")
    args = parser.parse_args()
    SkhSyncSelector(args.subject)
    plt.show()


if __name__ == "__main__":
    main()
