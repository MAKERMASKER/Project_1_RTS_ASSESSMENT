"""select_sync_zoom.py -- manual insole<->OpenCap sync for the trials
`decide_sync.py` flagged, with x-axis ZOOM for precise edge selection.

Same two-click convention as the existing GUIs (`select_lh_sync.py` /
`select_skh_sync.py`): one click on the OpenCap panel, one on the insole
panel, and `shift = opencap_click - insole_click`. Output CSV uses the
identical schema, so the existing promote scripts' format is preserved.

What is new
-----------
* **Scroll wheel zooms the x axis** about the cursor, on whichever panel the
  cursor is over. The two panels are on DIFFERENT clocks (OpenCap vs insole),
  so they are deliberately NOT x-linked -- each zooms independently.
  Shift+scroll pans. "Reset View" restores the full trace.
* The full trace is shown by default (the older GUIs clipped to the first 15 s),
  since zooming is now the way to get precision.
* **Foot PITCH is plotted alongside sole height.** Pitch was measured to be
  the far sharper touchdown cue for these hops -- ~0-5 deg planted against
  35-40 deg in flight, where the sole only clears 8-13 cm -- so it is the
  better thing to click against.
* **The automatic estimate is drawn as a dashed line** on both panels, and
  "Accept Auto" takes it in one click. Roughly 3 of the flagged trials were
  false alarms where the automatic answer was already right; those should be
  cheap to confirm rather than re-clicked from scratch.

Callbacks are wrapped in `_guarded` for the reason documented in
`select_lh_sync.py`: an uncaught exception inside a matplotlib/Tk callback can
leave the event loop's button grab stuck, which presents as a total freeze
rather than a visible crash.

Run from code/ (needs an interactive display):
    ..\\opensimPy\\Scripts\\python.exe expt_temp/state_machine_sync/select_sync_zoom.py --subject AB08
    ..\\opensimPy\\Scripts\\python.exe expt_temp/state_machine_sync/select_sync_zoom.py --list
"""
import argparse
import csv
import functools
import sys
import traceback
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent.parent
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(HERE))

from activities import activity_for  # noqa: E402
from config import DEFAULT  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from io_tables import read_insole, read_trc  # noqa: E402
from subject import Subject  # noqa: E402

from auto_sync import _insole_contacts, _runs  # noqa: E402
from foot_frame_sync import _foot_frame_signals  # noqa: E402

# `auto_sync` and `foot_frame_sync` are batch plotters: both call
# matplotlib.use("Agg") at import time. Importing them here would otherwise
# leave this GUI on a non-interactive canvas and the window never appears
# ("FigureCanvasAgg is non-interactive, and thus cannot be shown"). Restore an
# interactive backend AFTER those imports, before pyplot is bound.
matplotlib.use("TkAgg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.widgets import Button  # noqa: E402

RESULTS = HERE / "results"
DECISIONS = RESULTS / "sync_decisions.csv"
ZOOM_FACTOR = 1.35
FIELDNAMES = ["Trial", "Status", "OpenCapClickTimeS", "InsoleClickTimeS", "ShiftS"]


def _guarded(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception:
            print(f"[select_sync_zoom] error in {method.__name__} -- ignored, GUI stays open:")
            traceback.print_exc()
            return None
    return wrapper


def _flagged(subject_id=None):
    if not DECISIONS.is_file():
        raise SystemExit(f"No decisions file at {DECISIONS} -- run decide_sync.py first.")
    table = pd.read_csv(DECISIONS)
    table = table[table.Decision == "MANUAL"]
    if subject_id:
        table = table[table.Subject == subject_id]
    return table


class ZoomSyncSelector:
    def __init__(self, subject_id):
        self.subject_id = subject_id
        rows = _flagged(subject_id)
        if not len(rows):
            raise SystemExit(f"{subject_id}: no MANUAL-flagged trials.")

        discovery = TrialDiscovery(DEFAULT)
        subject = Subject(subject_id, DEFAULT)
        lookup = {}
        for activity_name in sorted(rows.Activity.unique()):
            activity = activity_for(activity_name, DEFAULT)
            for trial in discovery.trials(subject, activity, "all", for_analysis=True):
                lookup[trial.name] = trial

        self.queue = []
        for _index, row in rows.iterrows():
            trial = lookup.get(row.Trial)
            if trial is not None:
                self.queue.append((trial, float(row.AutoOffset_s) if np.isfinite(row.AutoOffset_s) else np.nan))
        if not self.queue:
            raise SystemExit(f"{subject_id}: flagged trials could not be resolved to Trial objects.")
        print(f"{subject_id}: {len(self.queue)} flagged trial(s) -> {[t.name for t, _o in self.queue]}")
        print("scroll = zoom x about cursor | shift+scroll = pan | click top then bottom")

        self.index = 0
        self.opencap_click = None
        self.insole_click = None
        self.decisions = {}
        self.markers = []
        self._redraw_timer = None

        self.figure, (self.top, self.bottom) = plt.subplots(2, 1, figsize=(15, 8))
        plt.subplots_adjust(bottom=0.20, hspace=0.42)
        self.pitch_axis = self.top.twinx()

        specs = [("Reset View", self._on_reset_view), ("Reset Clicks", self._on_reset_clicks),
                 ("Accept Auto", self._on_accept_auto), ("Reject && Next", self._on_reject),
                 ("Save && Next", self._on_save)]
        self.buttons = []
        left, width = 0.07, 0.17
        for label, handler in specs:
            axis = self.figure.add_axes([left, 0.05, width, 0.06])
            button = Button(axis, label)
            button.on_clicked(handler)
            self.buttons.append(button)
            left += width + 0.011

        self._status_text = self.figure.text(0.5, 0.005, "", ha="center", fontsize=9)
        self.figure.canvas.mpl_connect("button_press_event", self._on_click)
        self.figure.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.figure.canvas.mpl_connect("close_event", self._on_close)
        self._draw()

    def _schedule_redraw(self):
        """Coalesce a burst of scroll events into ONE redraw.

        A wheel flick emits 10-20 events; redrawing per event queued seconds
        of drawing in the Tk idle loop and the window stopped responding.
        Restarting a short single-shot timer means only the final view of a
        flick is ever painted. Falls back to an immediate draw if the backend
        has no timer (e.g. Agg under test).
        """
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

    def _status(self, message, colour="black"):
        """Every action gets visible feedback. A click that does nothing and
        says nothing is indistinguishable from a frozen GUI."""
        if getattr(self, "_status_text", None) is not None:
            self._status_text.set_text(message)
            self._status_text.set_color(colour)
            self.figure.canvas.draw_idle()

    @property
    def _current(self):
        return self.queue[self.index]

    # ---- drawing -----------------------------------------------------

    @_guarded
    def _draw(self):
        trial, auto_offset = self._current
        self.opencap_click = self.insole_click = None
        self.markers = []
        for axis in (self.top, self.pitch_axis, self.bottom):
            axis.clear()

        trc = read_trc(trial.effective_opencap_trc())
        insole = read_insole(trial.insole)
        side = trial.side if trial.side else "R"
        opencap_time = np.asarray(trc["time"], float)
        above, pitch, _speed = _foot_frame_signals(trc, side)

        self.top.plot(opencap_time, 100 * above, color="darkorange", lw=1.0,
                      label=f"sole height {side} (cm)")
        self.top.set_ylim(-2, 45)
        self.top.set_ylabel("sole height (cm)")
        self.pitch_axis.plot(opencap_time, pitch, color="purple", lw=0.9, alpha=0.85,
                             label=f"foot pitch {side} (deg)")
        self.pitch_axis.set_ylabel("pitch (deg)", color="purple")
        self.top.set_title(f"OpenCap -- click the touchdown (pitch drop is the sharpest cue)")

        insole_time = insole.time.to_numpy(float)
        column = "right_force" if side == "R" else "left_force"
        force = insole[column].to_numpy(float)
        mask, _events, _standing = _insole_contacts(force, insole_time)
        self.bottom.plot(insole_time, force, color="deepskyblue", lw=0.9, label=f"insole {side} (N)")
        # One axvspan per contact run rather than fill_between(where=...,
        # step="mid"). Measured: this is NOT the performance win it looks like
        # (0.367 s vs 0.369 s per redraw -- no difference; the ~0.3 s cost is
        # spread across rendering generally, not concentrated in any one
        # artist). Kept only because it reads more directly. The freeze was
        # fixed by throttling redraws in `_schedule_redraw`, not here.
        for start, end in _runs(mask):
            self.bottom.axvspan(insole_time[start], insole_time[end],
                                color="grey", alpha=0.15, lw=0)
        self.bottom.set_ylabel("insole force (N)")
        self.bottom.set_xlabel("insole clock (s)")
        self.bottom.set_title("Insole -- click the SAME touchdown")

        if np.isfinite(auto_offset):
            # Auto offset maps insole -> OpenCap; show the pair it implies,
            # anchored on the first detected insole contact.
            anchor_insole = insole_time[np.flatnonzero(mask)[0]] if mask.any() else insole_time[0]
            self.top.axvline(anchor_insole + auto_offset, color="green", ls="--", lw=1.1,
                             label=f"auto ({auto_offset:+.3f}s)")
            self.bottom.axvline(anchor_insole, color="green", ls="--", lw=1.1, label="auto")

        for axis in (self.top, self.bottom):
            axis.grid(True, alpha=0.3)
            axis.legend(fontsize=7, loc="upper right")
        self._full_view = (self.top.get_xlim(), self.bottom.get_xlim())

        self.figure.suptitle(
            f"[{self.index + 1}/{len(self.queue)}] {trial.name}  "
            f"(auto={auto_offset:+.3f}s -- verify or re-click)")
        self.figure.canvas.draw_idle()

    def _mark(self, axis, value, colour):
        self.markers.append(axis.axvline(value, color=colour, lw=1.4))
        self.figure.canvas.draw_idle()

    # ---- interaction -------------------------------------------------

    @_guarded
    def _on_scroll(self, event):
        if event.inaxes not in (self.top, self.pitch_axis, self.bottom) or event.xdata is None:
            return
        axis = self.bottom if event.inaxes is self.bottom else self.top
        low, high = axis.get_xlim()
        span = high - low
        modifiers = getattr(event, "modifiers", None) or ()
        if "shift" in modifiers:
            step = 0.2 * span * (1 if event.button == "up" else -1)
            axis.set_xlim(low + step, high + step)
        else:
            factor = 1.0 / ZOOM_FACTOR if event.button == "up" else ZOOM_FACTOR
            centre = event.xdata
            axis.set_xlim(centre - (centre - low) * factor, centre + (high - centre) * factor)
        self._schedule_redraw()

    def _clear_toolbar_mode(self):
        """Disengage the navigation toolbar's zoom/pan mode.

        While that mode is active matplotlib treats clicks as rubber-band
        drags, so a selection click must be ignored -- but ignoring it
        SILENTLY makes the GUI look permanently dead, because the mode is
        sticky and survives across trials. (That is exactly the "stopped
        accepting any clicks" failure: not an exception, just a swallowed
        click with no feedback.) So say so, and clear it, rather than
        dropping the click into a hole.
        """
        toolbar = self.figure.canvas.toolbar
        mode = str(getattr(toolbar, "mode", "") or "")
        if not mode:
            return False
        try:
            if "zoom" in mode:
                toolbar.zoom()
            elif "pan" in mode:
                toolbar.pan()
        except Exception:  # noqa: BLE001 -- clearing is best-effort
            traceback.print_exc()
        print(f"  [toolbar '{mode}' was active -- clicks were being swallowed; "
              "cleared it, click again]")
        self._status(f"toolbar '{mode}' cleared -- click again", "red")
        return True

    @_guarded
    def _on_click(self, event):
        if event.inaxes is None or event.button != 1:
            return
        if self._clear_toolbar_mode():
            return
        if event.inaxes in (self.top, self.pitch_axis):
            self.opencap_click = float(event.xdata)
            self._mark(self.top, self.opencap_click, "red")
            print(f"  opencap click {self.opencap_click:.4f}s")
            self._status(f"OpenCap click {self.opencap_click:.4f}s", "green")
        elif event.inaxes is self.bottom:
            self.insole_click = float(event.xdata)
            self._mark(self.bottom, self.insole_click, "red")
            print(f"  insole click {self.insole_click:.4f}s")
            self._status(f"insole click {self.insole_click:.4f}s", "green")

    @_guarded
    def _on_reset_view(self, _event):
        top_view, bottom_view = self._full_view
        self.top.set_xlim(*top_view)
        self.bottom.set_xlim(*bottom_view)
        self.figure.canvas.draw_idle()

    @_guarded
    def _on_reset_clicks(self, _event):
        for marker in self.markers:
            try:
                marker.remove()
            except Exception:  # noqa: BLE001
                pass
        self.markers = []
        self.opencap_click = self.insole_click = None
        self.figure.canvas.draw_idle()

    @_guarded
    def _record(self, status, opencap, insole):
        trial, _auto = self._current
        shift = None if opencap is None or insole is None else opencap - insole
        self.decisions[trial.name] = {
            "Trial": trial.name, "Status": status,
            "OpenCapClickTimeS": opencap, "InsoleClickTimeS": insole, "ShiftS": shift,
        }
        print(f"  {trial.name}: {status}" + (f", shift={shift:.4f}s" if shift is not None else ""))
        self._advance()

    @_guarded
    def _on_accept_auto(self, _event):
        trial, auto_offset = self._current
        if not np.isfinite(auto_offset):
            print("  (no automatic estimate for this trial -- click manually)")
            return
        # Record the automatic offset in click form so downstream stays uniform.
        self._record("synced", float(auto_offset), 0.0)

    @_guarded
    def _on_reject(self, _event):
        self._record("rejected", None, None)

    @_guarded
    def _on_save(self, _event):
        if self.opencap_click is None or self.insole_click is None:
            print("  (need one click on each panel before saving -- ignored)")
            self._status("need one click on EACH panel before saving", "red")
            return
        self._record("synced", self.opencap_click, self.insole_click)

    @_guarded
    def _advance(self):
        if self.index + 1 < len(self.queue):
            self.index += 1
            self._draw()
        else:
            print(f"{self.subject_id}: last trial decided -- close the window to save.")

    @_guarded
    def _on_close(self, _event):
        if not self.decisions:
            print(f"{self.subject_id}: nothing decided, nothing saved.")
            return
        path = RESULTS / f"{self.subject_id}_manual_sync.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.decisions.values())
        synced = sum(1 for r in self.decisions.values() if r["Status"] == "synced")
        print(f"{self.subject_id}: {len(self.decisions)} decided -> {path} "
              f"({synced} synced, {len(self.decisions) - synced} rejected)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject")
    parser.add_argument("--list", action="store_true", help="print the manual queue and exit")
    args = parser.parse_args()
    if args.list or not args.subject:
        table = _flagged(args.subject)
        print(f"{len(table)} flagged trial(s):")
        for _index, row in table.iterrows():
            print(f"  {row.Subject:<7} {row.Activity:<13} {row.Trial:<22} "
                  f"auto={row.AutoOffset_s} spread={row.EstimateSpread_s}")
        if not args.subject:
            print("\nPass --subject <ID> to open the GUI.")
        return
    backend = matplotlib.get_backend()
    if backend.lower().startswith("agg"):
        raise SystemExit(
            f"matplotlib is on the non-interactive '{backend}' backend, so no window can "
            "open. Something imported before this line forced it; check for a "
            "matplotlib.use('Agg') in an imported module."
        )
    ZoomSyncSelector(args.subject)
    plt.show()


if __name__ == "__main__":
    main()
