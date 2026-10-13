"""review_sync.py -- fast pass/reject review of the insole<->OpenCap sync for
every subject and activity, with the option to re-set a sync by eye.

Produces a validated list of trials whose sync has been confirmed by a human,
which is the gate everything downstream should respect.

Scope: AB03/AB05/AB07/AB08/AB10/NTS01 x single_leg_jump / lateral_hop /
skater_hop / triple_hop / walk = 235 trials. The 505 trials drop out on their
own because no `Activity` subclass claims them; AB11 is excluded until its
processing is set up.

Flow
----
Default REVIEW view puts everything on the OpenCap clock: OpenCap foot
kinematics on top, insole (shifted by the trial's current sync) below, with
the force plate overlaid as an independent check. If the sync is right the
contacts line up vertically -- so the common case is a glance and a keypress.

  p / Pass      sync looks right
  r / Reject    sync is wrong or the trial is unusable
  f / Fix       switch to two-panel click mode (independent clocks) and
                re-set the shift: click OpenCap, then insole, then Apply
  b / Back      previous trial
  n / Next      skip without deciding

Every decision is written to disk IMMEDIATELY. The earlier sync GUI only
saved on window close and a freeze lost two trials' work; at 235 trials that
risk is not acceptable.

GUI robustness -- all of these were real failures in the earlier GUI:
  * `auto_sync`/`foot_frame_sync` call matplotlib.use("Agg") at import, which
    leaves the window unable to open. The backend is restored AFTER those
    imports.
  * Scroll-zoom redraws are throttled; redrawing per wheel notch queued
    seconds of drawing and the window stopped responding.
  * Every callback is guarded, so an exception cannot wedge the Tk event loop.
  * An active navigation-toolbar mode silently swallows clicks, so it is
    detected, reported and cleared instead of dropping the click.

Run from code/ (needs an interactive display):
    ..\\opensimPy\\Scripts\\python.exe expt_temp/sync_review_gui/review_sync.py
    ..\\opensimPy\\Scripts\\python.exe expt_temp/sync_review_gui/review_sync.py --subject AB08
    ..\\opensimPy\\Scripts\\python.exe expt_temp/sync_review_gui/review_sync.py --redo   (re-review decided trials)
"""
import argparse
import csv
import functools
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent.parent
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "expt_temp" / "state_machine_sync"))

from activities import activity_for  # noqa: E402
from config import DEFAULT  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from insole_geometry import InsoleGeometry  # noqa: E402
from io_tables import read_insole, read_storage, read_trc  # noqa: E402
from subject import Subject  # noqa: E402

# Restore an interactive backend AFTER any import that may have forced Agg.
matplotlib.use("TkAgg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.widgets import Button  # noqa: E402

SUBJECT_IDS = ("AB03", "AB05", "AB07", "AB08", "AB10", "NTS01")
ACTIVITIES = ("single_leg_jump", "lateral_hop", "skater_hop", "triple_hop", "walk")
RESULTS = HERE / "results"
DECISIONS = RESULTS / "sync_review.csv"
FIELDNAMES = ["Subject", "Activity", "Trial", "Decision", "Shift_s", "ShiftSource",
              "Reviewed_at", "Note"]
ZOOM_FACTOR = 1.35


def _thin(x, y, limit=3000):
    """Decimate a trace for DISPLAY only.

    Drawing every sample cost ~0.5 s per redraw, and a redraw happens on every
    advance, zoom and click. At screen resolution a few thousand points is
    visually identical. The underlying data is never modified -- only what is
    handed to the renderer.
    """
    n = len(x)
    if n <= limit:
        return x, y
    step = int(np.ceil(n / limit))
    return x[::step], y[::step]


def _guarded(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception:
            print(f"[review_sync] error in {method.__name__} -- ignored, GUI stays open:")
            traceback.print_exc()
            try:
                self._status(f"error in {method.__name__} (see console)", "red")
            except Exception:  # noqa: BLE001
                pass
            return None
    return wrapper


def _load_decisions():
    """Existing decisions keyed by trial name, so a review can be resumed."""
    if not DECISIONS.is_file():
        return {}
    return {record["Trial"]: record
            for record in pd.read_csv(DECISIONS).to_dict("records")}


def _plate_offset(time, reference, plate_time, plate_values):
    """Cross-correlation offset so the plate can be overlaid for validation."""
    from scipy import signal as scipy_signal
    dt = float(np.median(np.diff(time)))
    grid = np.arange(float(plate_time.min()), float(plate_time.max()), dt)
    resampled = np.interp(grid, plate_time, plate_values)
    a = resampled - resampled.mean()
    b = np.nan_to_num(reference - np.nanmean(reference))
    if len(a) < 4 or len(b) < 4:
        return np.nan
    correlation = scipy_signal.correlate(a, b, mode="full")
    lags = scipy_signal.correlation_lags(len(a), len(b), mode="full")
    best = int(lags[int(np.argmax(correlation))])
    return float(grid[0] + best * dt - time[0])


class SyncReviewer:
    def __init__(self, subject_ids, activities, redo=False):
        discovery = TrialDiscovery(DEFAULT)
        self.decisions = _load_decisions()
        self.queue = []
        for subject_id in subject_ids:
            subject = Subject(subject_id, DEFAULT)
            for activity_name in activities:
                activity = activity_for(activity_name, DEFAULT)
                for trial in sorted(discovery.trials(subject, activity, "all", for_analysis=True),
                                    key=lambda t: t.name):
                    if not redo and trial.name in self.decisions:
                        continue
                    self.queue.append((subject, activity, trial))
        if not self.queue:
            raise SystemExit("Nothing to review (all trials already decided -- pass --redo).")
        print(f"{len(self.queue)} trial(s) to review "
              f"({len(self.decisions)} already decided)")
        print("keys:  p=pass  r=reject  f=fix sync  b=back  n=skip   scroll=zoom")

        self.index = 0
        self.mode = "review"
        self.opencap_click = None
        self.insole_click = None
        self._redraw_timer = None
        self._cache = {}
        self._busy = False

        self.figure, self.axes = plt.subplots(2, 1, figsize=(16, 8))
        plt.subplots_adjust(bottom=0.17, hspace=0.33)
        self.pitch_axis = self.axes[0].twinx()

        specs = [("Pass (p)", self._on_pass), ("Reject (r)", self._on_reject),
                 ("Fix sync (f)", self._on_fix), ("Apply fix", self._on_apply),
                 ("Back (b)", self._on_back), ("Skip (n)", self._on_skip)]
        self.buttons = []
        left, width = 0.04, 0.145
        for label, handler in specs:
            axis = self.figure.add_axes([left, 0.04, width, 0.055])
            button = Button(axis, label)
            button.on_clicked(handler)
            self.buttons.append(button)
            left += width + 0.012

        self._status_text = self.figure.text(0.5, 0.003, "", ha="center", fontsize=9)
        canvas = self.figure.canvas
        canvas.mpl_connect("button_press_event", self._on_click)
        canvas.mpl_connect("scroll_event", self._on_scroll)
        canvas.mpl_connect("key_press_event", self._on_key)
        canvas.mpl_connect("close_event", lambda _e: print("closed -- decisions already saved"))
        self._draw()

    # ---- infrastructure ---------------------------------------------

    def _status(self, message, colour="black", redraw=True):
        """Set the status line. `redraw=False` when the caller is about to
        render anyway -- otherwise _draw() renders the whole figure TWICE
        (measured: 1.14 s instead of 0.6 s per advance, which is what backed
        up the Tk event queue and made the window look frozen)."""
        self._status_text.set_text(message)
        self._status_text.set_color(colour)
        if redraw:
            self.figure.canvas.draw_idle()

    def _schedule_redraw(self):
        canvas = self.figure.canvas
        if not hasattr(canvas, "new_timer"):
            canvas.draw_idle()
            return
        if self._redraw_timer is None:
            self._redraw_timer = canvas.new_timer(interval=70)
            self._redraw_timer.single_shot = True
            self._redraw_timer.add_callback(lambda: self.figure.canvas.draw_idle())
        self._redraw_timer.stop()
        self._redraw_timer.start()

    def _clear_toolbar_mode(self):
        toolbar = self.figure.canvas.toolbar
        mode = str(getattr(toolbar, "mode", "") or "")
        if not mode:
            return False
        try:
            if "zoom" in mode:
                toolbar.zoom()
            elif "pan" in mode:
                toolbar.pan()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        self._status(f"toolbar '{mode}' cleared -- click again", "red")
        return True

    @property
    def _current(self):
        return self.queue[self.index]

    # ---- data --------------------------------------------------------

    def _trial_data(self):
        return self._load(*self._current)

    def _load(self, subject, activity, trial):
        if trial.name in self._cache:
            return self._cache[trial.name]
        record = {"error": None}
        try:
            trc = read_trc(trial.effective_opencap_trc())
            insole = read_insole(trial.insole)
            side = trial.side if trial.side else "R"
            record["side"] = side
            record["opencap_time"] = np.asarray(trc["time"], float)
            heel = InsoleGeometry.marker(trc, record["opencap_time"], side, "heel")
            toe1 = InsoleGeometry.marker(trc, record["opencap_time"], side, "toe1")
            toe5 = InsoleGeometry.marker(trc, record["opencap_time"], side, "toe5")
            _origin, rotation = InsoleGeometry.frames(trc, record["opencap_time"], side)
            vertical = rotation[:, :, 2]
            lowest = np.minimum((heel - 0.03 * vertical)[:, 1],
                                (0.5 * (toe1 + toe5) - 0.03 * vertical)[:, 1])
            ground = np.nanpercentile(lowest[np.isfinite(lowest)], 5)
            record["sole_height"] = lowest - ground
            record["pitch"] = np.degrees(np.arccos(np.clip(vertical[:, 1], -1, 1)))
            record["insole_time"] = insole.time.to_numpy(float)
            record["insole_force"] = insole[
                "right_force" if side == "R" else "left_force"].to_numpy(float)
            try:
                _t, _i, _k, sync = activity.synchronize(
                    trial, trc, read_storage(trial.effective_opencap_ik()), insole, subject.rules)
                record["shift"] = float(sync["shift_s"])
                record["shift_source"] = sync.get("method", "activity.synchronize")
                record["quality_omit"] = bool(sync.get("quality_omit", False))
            except Exception as error:  # noqa: BLE001
                record["shift"] = np.nan
                record["shift_source"] = f"UNAVAILABLE: {type(error).__name__}"
                record["quality_omit"] = True
                record["error"] = str(error)
            record["plate"] = None
            if trial.force_mot and Path(trial.force_mot).is_file():
                force_plate = read_storage(trial.force_mot)
                number = 1 if side == "R" else 2
                column = f"ground_force_{number}_vy"
                if column in force_plate:
                    record["plate_time"] = force_plate.time.to_numpy(float)
                    record["plate"] = np.abs(force_plate[column].to_numpy(float))
        except Exception as error:  # noqa: BLE001
            record["error"] = f"{type(error).__name__}: {error}"
        self._cache[trial.name] = record
        if len(self._cache) > 6:
            self._cache.pop(next(iter(self._cache)))
        return record

    def _prefetch_next(self):
        """Warm the NEXT trial's cache on a background thread.

        Loading cost ~1.4 s per trial, and doing it inside the key/button
        callback blocked the Tk event loop for long enough that queued input
        piled up and the window looked frozen. The user spends seconds looking
        at each trial, which is ample time to load the next one. The thread
        only reads files and fills a dict -- it never touches matplotlib.
        """
        if self.index + 1 >= len(self.queue):
            return
        nxt = self.queue[self.index + 1]
        if nxt[2].name in self._cache:
            return

        def work():
            try:
                self._load(*nxt)
            except Exception:  # noqa: BLE001 -- prefetch is best-effort
                pass
        threading.Thread(target=work, daemon=True).start()

    def _focus_canvas(self):
        """Return keyboard focus to the canvas after a button click, so the
        p/r/f/b/n keys keep working (a Tk Button steals focus otherwise)."""
        try:
            self.figure.canvas.get_tk_widget().focus_set()
        except Exception:  # noqa: BLE001 -- not fatal, and backend-specific
            pass

    # ---- drawing -----------------------------------------------------

    @_guarded
    def _draw(self):
        self._busy = True
        try:
            self._draw_inner()
        finally:
            self._busy = False

    def _draw_inner(self):
        subject, activity, trial = self._current
        data = self._trial_data()
        for axis in (self.axes[0], self.axes[1], self.pitch_axis):
            axis.clear()
        self.opencap_click = self.insole_click = None

        decided = self.decisions.get(trial.name)
        header = (f"[{self.index + 1}/{len(self.queue)}]  {trial.name}   "
                  f"({activity.name}, side {data.get('side', '?')})")
        if data.get("error"):
            self.axes[0].text(0.5, 0.5, f"LOAD FAILED\n{data['error']}", ha="center",
                              va="center", transform=self.axes[0].transAxes, color="red")
            self.figure.suptitle(header + "   -- cannot display", color="red")
            self.figure.canvas.draw_idle()
            return

        shift = data["shift"]
        if self.mode == "review" and np.isfinite(shift):
            # Everything on the OpenCap clock: aligned contacts == good sync.
            self.axes[0].plot(*_thin(data["opencap_time"], 100 * data["sole_height"]),
                              color="darkorange", lw=1.0, label="sole height (cm)")
            self.axes[0].set_ylim(-2, 45)
            self.axes[0].set_ylabel("sole height (cm)")
            self.pitch_axis.plot(*_thin(data["opencap_time"], data["pitch"]), color="purple",
                                 lw=0.9, alpha=0.8, label="foot pitch (deg)")
            self.pitch_axis.set_ylabel("pitch (deg)", color="purple")
            self.axes[0].set_title("OpenCap kinematics")

            shifted = data["insole_time"] + shift
            self.axes[1].plot(*_thin(shifted, data["insole_force"]), color="deepskyblue", lw=1.0,
                              label="insole (shifted by current sync)")
            if data["plate"] is not None:
                aligned = np.interp(data["opencap_time"], shifted, data["insole_force"],
                                    left=np.nan, right=np.nan)
                offset = _plate_offset(data["opencap_time"], aligned,
                                       data["plate_time"], data["plate"])
                if np.isfinite(offset):
                    self.axes[1].plot(data["opencap_time"],
                                      np.interp(data["opencap_time"] + offset,
                                                data["plate_time"], data["plate"],
                                                left=np.nan, right=np.nan),
                                      color="black", lw=1.0, alpha=0.75,
                                      label="force plate (independent check)")
            self.axes[1].set_xlabel("OpenCap clock (s)   -- contacts should line up with the panel above")
            self.axes[1].set_ylabel("force (N)")
            low = max(float(data["opencap_time"].min()), float(shifted.min()))
            high = min(float(data["opencap_time"].max()), float(shifted.max()))
            for axis in self.axes:
                axis.set_xlim(low, high)
        else:
            # FIX mode (or no usable shift): independent clocks for clicking.
            self.axes[0].plot(*_thin(data["opencap_time"], 100 * data["sole_height"]),
                              color="darkorange", lw=1.0, label="sole height (cm)")
            self.axes[0].set_ylim(-2, 45)
            self.pitch_axis.plot(*_thin(data["opencap_time"], data["pitch"]), color="purple",
                                 lw=0.9, alpha=0.8, label="foot pitch (deg)")
            self.axes[0].set_title("OpenCap clock -- click the touchdown (pitch drop is sharpest)")
            self.axes[1].plot(*_thin(data["insole_time"], data["insole_force"]),
                              color="deepskyblue", lw=1.0, label="insole")
            self.axes[1].set_title("Insole clock -- click the SAME touchdown")
            self.axes[1].set_xlabel("insole clock (s)")
        self._full_view = (self.axes[0].get_xlim(), self.axes[1].get_xlim())

        for axis in self.axes:
            axis.grid(True, alpha=0.3)
            axis.legend(fontsize=7, loc="upper right")

        flags = []
        if data.get("quality_omit"):
            flags.append("quality_omit")
        if not np.isfinite(shift):
            flags.append(data["shift_source"])
        suffix = f"   shift={shift:+.3f}s [{data['shift_source'][:30]}]" if np.isfinite(shift) else ""
        if decided:
            suffix += f"   ALREADY: {decided['Decision']}"
        colour = "red" if flags else "black"
        self.figure.suptitle(header + suffix + ("   !! " + ", ".join(flags) if flags else ""),
                             color=colour)
        self._status(f"mode={self.mode}", "black", redraw=False)
        self.figure.canvas.draw_idle()
        self._focus_canvas()
        self._prefetch_next()

    # ---- interaction -------------------------------------------------

    @_guarded
    def _on_scroll(self, event):
        if event.inaxes is None or event.xdata is None:
            return
        axis = self.axes[1] if event.inaxes is self.axes[1] else self.axes[0]
        low, high = axis.get_xlim()
        modifiers = getattr(event, "modifiers", None) or ()
        if "shift" in modifiers:
            step = 0.2 * (high - low) * (1 if event.button == "up" else -1)
            axis.set_xlim(low + step, high + step)
        else:
            factor = 1.0 / ZOOM_FACTOR if event.button == "up" else ZOOM_FACTOR
            centre = event.xdata
            axis.set_xlim(centre - (centre - low) * factor, centre + (high - centre) * factor)
        if self.mode == "review":   # shared clock -- keep both panels together
            self.axes[0].set_xlim(*axis.get_xlim())
            self.axes[1].set_xlim(*axis.get_xlim())
        self._schedule_redraw()

    @_guarded
    def _on_click(self, event):
        if event.inaxes is None or event.button != 1:
            return
        if self._clear_toolbar_mode():
            return
        if self.mode != "fix":
            self._status("press 'f' to fix the sync before clicking", "red")
            return
        if event.inaxes in (self.axes[0], self.pitch_axis):
            self.opencap_click = float(event.xdata)
            self.axes[0].axvline(self.opencap_click, color="red", lw=1.4)
            self._status(f"OpenCap click {self.opencap_click:.4f}s", "green")
        elif event.inaxes is self.axes[1]:
            self.insole_click = float(event.xdata)
            self.axes[1].axvline(self.insole_click, color="red", lw=1.4)
            self._status(f"insole click {self.insole_click:.4f}s", "green")
        self.figure.canvas.draw_idle()

    @_guarded
    def _on_key(self, event):
        if self._busy:
            # A redraw costs ~0.65 s. Without this, keys pressed during one
            # queue up and each triggers another redraw, so the backlog grows
            # faster than it drains and the window appears permanently frozen.
            return
        {"p": self._on_pass, "r": self._on_reject, "f": self._on_fix,
         "b": self._on_back, "n": self._on_skip}.get(event.key, lambda _e: None)(event)

    @_guarded
    def _on_fix(self, _event):
        self.mode = "fix"
        self._draw()
        self._status("FIX mode: click OpenCap panel, then insole panel, then 'Apply fix'", "blue")

    @_guarded
    def _on_apply(self, _event):
        if self.opencap_click is None or self.insole_click is None:
            self._status("need one click on EACH panel first", "red")
            return
        shift = self.opencap_click - self.insole_click
        self._trial_data()["shift"] = shift
        self._trial_data()["shift_source"] = "manual_review_gui"
        self.mode = "review"
        self._draw()
        self._status(f"shift set to {shift:+.4f}s -- check alignment, then Pass", "green")

    def _record(self, decision, note=""):
        subject, activity, trial = self._current
        data = self._trial_data()
        self.decisions[trial.name] = {
            "Subject": subject.id, "Activity": activity.name, "Trial": trial.name,
            "Decision": decision,
            "Shift_s": round(data.get("shift", np.nan), 5) if np.isfinite(data.get("shift", np.nan)) else "",
            "ShiftSource": data.get("shift_source", ""),
            "Reviewed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Note": note,
        }
        self._save()
        print(f"  {trial.name}: {decision}")
        self._advance()

    def _save(self):
        RESULTS.mkdir(parents=True, exist_ok=True)
        with DECISIONS.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.decisions.values())

    @_guarded
    def _on_pass(self, _event):
        self._record("PASS")

    @_guarded
    def _on_reject(self, _event):
        self._record("REJECT")

    @_guarded
    def _on_skip(self, _event):
        self._advance()

    @_guarded
    def _on_back(self, _event):
        if self.index > 0:
            self.index -= 1
            self.mode = "review"
            self._draw()

    @_guarded
    def _advance(self):
        self.mode = "review"
        if self.index + 1 < len(self.queue):
            self.index += 1
            self._draw()
        else:
            passed = sum(1 for d in self.decisions.values() if d["Decision"] == "PASS")
            print(f"\nReached the end. {len(self.decisions)} decided, {passed} PASS.")
            print(f"-> {DECISIONS}")
            self._status("end of queue -- close the window", "blue")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject")
    parser.add_argument("--activity")
    parser.add_argument("--redo", action="store_true",
                        help="include trials that already have a decision")
    args = parser.parse_args()
    if matplotlib.get_backend().lower().startswith("agg"):
        raise SystemExit(f"matplotlib is on the non-interactive '{matplotlib.get_backend()}' "
                         "backend -- no window can open.")
    SyncReviewer((args.subject,) if args.subject else SUBJECT_IDS,
                 (args.activity,) if args.activity else ACTIVITIES,
                 redo=args.redo)
    plt.show()


if __name__ == "__main__":
    main()
