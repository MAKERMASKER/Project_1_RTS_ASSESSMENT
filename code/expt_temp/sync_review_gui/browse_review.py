"""browse_review.py -- pass/reject browser over the pre-rendered sync plots.

This exists because the interactive version froze three times. Everything that
could block the Tk event loop has been moved out: no OpenSim calls, no file
parsing, no `activity.synchronize()`, no background threads, no scipy. Each
advance loads one PNG and blits it, which is milliseconds.

Run `render_review_plots.py` first.

  p / Pass      sync looks right
  r / Reject    sync wrong or trial unusable
  u / Unsure    park it for a second look
  b / Back      previous
  n / Next      skip without deciding

Decisions are written to disk on every keypress (results/sync_review.csv) and
the queue resumes where it left off.

Run from code/ (needs an interactive display):
    ..\\opensimPy\\Scripts\\python.exe expt_temp/sync_review_gui/browse_review.py
    ..\\opensimPy\\Scripts\\python.exe expt_temp/sync_review_gui/browse_review.py --redo
"""
import argparse
import csv
import functools
import sys
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("TkAgg", force=True)
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.widgets import Button  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
INDEX = RESULTS / "render_index.csv"
DECISIONS = RESULTS / "sync_review.csv"
FIELDNAMES = ["Subject", "Activity", "Trial", "Decision", "Shift_s", "ShiftSource",
              "QualityOmit", "Reviewed_at"]


def _guarded(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception:
            print(f"[browse_review] error in {method.__name__}:")
            traceback.print_exc()
            return None
    return wrapper


class Browser:
    def __init__(self, rows, decisions):
        self.rows = rows
        self.decisions = decisions
        self.index = 0
        self.figure = plt.figure(figsize=(16, 9))
        self.axis = self.figure.add_axes([0.0, 0.08, 1.0, 0.88])
        self.axis.axis("off")
        self._image = None

        specs = [("Pass (p)", self._on_pass), ("Reject (r)", self._on_reject),
                 ("Unsure (u)", self._on_unsure), ("Back (b)", self._on_back),
                 ("Skip (n)", self._on_skip)]
        self.buttons = []
        left, width = 0.06, 0.16
        for label, handler in specs:
            axis = self.figure.add_axes([left, 0.015, width, 0.045])
            button = Button(axis, label)
            button.on_clicked(handler)
            self.buttons.append(button)
            left += width + 0.015

        canvas = self.figure.canvas
        canvas.mpl_connect("key_press_event", self._on_key)
        canvas.mpl_connect("close_event", lambda _e: self._summary())
        self._draw()

    def _focus(self):
        try:
            self.figure.canvas.get_tk_widget().focus_set()
        except Exception:  # noqa: BLE001
            pass

    @property
    def _current(self):
        return self.rows[self.index]

    @_guarded
    def _draw(self):
        row = self._current
        self.axis.clear()
        self.axis.axis("off")
        path = HERE / str(row.get("Plot", ""))
        if row.get("Plot") and path.is_file():
            self.axis.imshow(mpimg.imread(path))
        else:
            self.axis.text(0.5, 0.5, f"NO PLOT\n{row['Trial']}\n{row.get('Error', '')}",
                           ha="center", va="center", fontsize=13, color="red")
        decided = self.decisions.get(row["Trial"], {}).get("Decision", "")
        counts = pd.Series([d["Decision"] for d in self.decisions.values()]).value_counts().to_dict() \
            if self.decisions else {}
        self.figure.suptitle(
            f"[{self.index + 1}/{len(self.rows)}]  {row['Trial']}"
            + (f"   ALREADY: {decided}" if decided else "")
            + f"      decided so far: {counts}",
            fontsize=10)
        self.figure.canvas.draw_idle()
        self._focus()

    def _record(self, decision):
        row = self._current
        self.decisions[row["Trial"]] = {
            "Subject": row["Subject"], "Activity": row["Activity"], "Trial": row["Trial"],
            "Decision": decision, "Shift_s": row.get("Shift_s", ""),
            "ShiftSource": row.get("ShiftSource", ""),
            "QualityOmit": row.get("QualityOmit", ""),
            "Reviewed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        RESULTS.mkdir(parents=True, exist_ok=True)
        with DECISIONS.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.decisions.values())
        print(f"  {row['Trial']}: {decision}")
        self._advance()

    @_guarded
    def _on_key(self, event):
        {"p": self._on_pass, "r": self._on_reject, "u": self._on_unsure,
         "b": self._on_back, "n": self._on_skip}.get(event.key, lambda _e: None)(event)

    @_guarded
    def _on_pass(self, _event):
        self._record("PASS")

    @_guarded
    def _on_reject(self, _event):
        self._record("REJECT")

    @_guarded
    def _on_unsure(self, _event):
        self._record("UNSURE")

    @_guarded
    def _on_skip(self, _event):
        self._advance()

    @_guarded
    def _on_back(self, _event):
        if self.index > 0:
            self.index -= 1
            self._draw()

    @_guarded
    def _advance(self):
        if self.index + 1 < len(self.rows):
            self.index += 1
            self._draw()
        else:
            self._summary()

    def _summary(self):
        if not self.decisions:
            return
        counts = pd.Series([d["Decision"] for d in self.decisions.values()]).value_counts()
        print(f"\n{len(self.decisions)} decided: {counts.to_dict()}")
        print(f"-> {DECISIONS}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject")
    parser.add_argument("--activity")
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()
    if not INDEX.is_file():
        raise SystemExit(f"No {INDEX} -- run render_review_plots.py first.")
    if matplotlib.get_backend().lower().startswith("agg"):
        raise SystemExit("matplotlib is on a non-interactive backend; no window can open.")

    rows = pd.read_csv(INDEX).to_dict("records")
    if args.subject:
        rows = [r for r in rows if r["Subject"] == args.subject]
    if args.activity:
        rows = [r for r in rows if r["Activity"] == args.activity]
    decisions = ({r["Trial"]: r for r in pd.read_csv(DECISIONS).to_dict("records")}
                 if DECISIONS.is_file() else {})
    if not args.redo:
        rows = [r for r in rows if r["Trial"] not in decisions]
    if not rows:
        raise SystemExit("Nothing to review (all decided -- pass --redo).")
    print(f"{len(rows)} to review, {len(decisions)} already decided")
    print("keys: p=pass  r=reject  u=unsure  b=back  n=skip")
    Browser(rows, decisions)
    plt.show()


if __name__ == "__main__":
    main()
