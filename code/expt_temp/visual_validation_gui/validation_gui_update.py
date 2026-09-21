"""validation_gui_update.py -- fast, 2D-only marker-vs-framework Force/CoP
browser across all subjects/trials.

Why this exists alongside validation_gui.py (kept, unchanged): that tool's
3D skeleton view needs an OpenSim `Model` built and driven through
per-frame forward kinematics for every trial, which is slow and (per real
usage) makes trial switching feel unresponsive when navigating quickly
across many subjects/trials. But the actual need -- reviewing Force/CoP
agreement between marker and framework across ~70 trials -- doesn't require
a skeleton at all: the Force/CoP data is already fully computed as plain
numpy/pandas data (trial_data.npz, the marker force-plate .mot file), so
displaying it needs no OpenSim Model construction and no threading. This
tool trades the 3D view for near-instant trial switching; when you actually
want the 3D skeleton for one specific trial, use the "Open in OpenSim"
buttons (reusing validation_gui.py's already-reliable native_play(), a
single blocking native visualizer window, no custom decoration logic).

Also makes explicit what was previously easy to misread as a bug: the
framework/insole pipeline only ever computes data within the SLJ activity's
own +/-0.75s analysis window (activities.py's synchronize() never produces
anything outside it) -- there is no "missing" wider framework signal to
show. The marker/force-plate recording, by contrast, spans the whole raw
trial, so this tool plots that fuller marker context (thin gray line)
alongside a shaded band marking the analysis window, with both marker and
framework shown at full color/weight only inside that shaded band -- so the
window is visually obvious rather than implicit.

Run from the opensimPy environment, from this folder:
    python validation_gui_update.py
"""
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
import numpy as np  # noqa: E402

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from activities import activity_for  # noqa: E402
from config import DEFAULT  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from insole_geometry import InsoleGeometry  # noqa: E402
from io_tables import read_storage, read_trc  # noqa: E402
from subject import Subject  # noqa: E402

from validation_gui import native_play  # noqa: E402  (reuse the one reliable native-viewer path)

SUBJECT_IDS = ("AB03", "AB05", "AB07", "AB08", "NTS01")
ACTIVITY_NAME = "single_leg_jump"


def load_trial_forces(trial):
    """Force/CoP-only load: no OpenSim Model, no forward kinematics -- just
    the numpy/pandas data the production pipeline already computed. This is
    what makes trial switching near-instant (milliseconds, not seconds)."""
    package_path = trial.output / "trial_data.npz"
    if not package_path.is_file():
        raise FileNotFoundError(f"No framework output for {trial.name}: {package_path}")
    package = np.load(package_path, allow_pickle=True)
    window_time = package["time"].astype(float)  # framework's own +/-0.75s analysis window
    side = "right" if trial.side == "R" else "left"
    framework_force = package[f"{side}_grf_N"].astype(float)
    framework_cop = package[f"{side}_cop_m"].astype(float).copy()
    framework_contact = package[f"{side}_contact"].astype(bool)
    # trial_data.npz stores 0 (not NaN) for CoP outside ground contact -- fine
    # for force (0 N outside contact is physically correct), but a CoP of
    # (0, 0, 0) during flight is not a real position and draws a spurious
    # point/line if plotted as-is. NaN it out for display, same as
    # validation_gui.py already does.
    framework_cop[~framework_contact] = np.nan

    force_plate = read_storage(trial.force_mot)  # full raw trial duration
    plate = 1 if trial.side == "R" else 2
    marker_vertical = np.abs(force_plate[f"ground_force_{plate}_vy"].to_numpy(float))
    # Same peak-alignment shift outcomes.py.analyze_trial() uses -- marker
    # mocap is its own independent recording, not on the framework's clock.
    shift = float(window_time[np.argmax(package["force"])] - force_plate["time"].iloc[np.argmax(marker_vertical)])

    full_marker_time = force_plate["time"].to_numpy(float) + shift  # now on the framework's clock
    full_marker_force = force_plate[[f"ground_force_{plate}_v{axis}" for axis in "xyz"]].to_numpy(float)
    full_marker_cop = force_plate[[f"ground_force_{plate}_p{axis}" for axis in "xyz"]].to_numpy(float)
    if np.nanmedian(np.abs(full_marker_cop)) > 2:
        full_marker_cop = full_marker_cop / 1000
    contact = np.abs(full_marker_force[:, 1]) >= 25
    full_marker_cop = full_marker_cop.copy()
    full_marker_cop[~contact] = np.nan

    window_marker_force = np.column_stack([
        np.interp(window_time, full_marker_time, full_marker_force[:, axis], left=np.nan, right=np.nan)
        for axis in range(3)
    ])
    window_marker_cop = np.column_stack([
        np.interp(window_time, full_marker_time, full_marker_cop[:, axis], left=np.nan, right=np.nan)
        for axis in range(3)
    ])

    # Marker (force-plate) and framework (insole) CoP are each reported in
    # their OWN system's absolute lab frame, which use different origins --
    # raw ground-frame values from the two sources are not directly
    # comparable (this is a known, pre-existing difference between the two
    # recording systems, not a sync bug). validation_gui.py's 3D tool
    # already handles this correctly by converting both into a common
    # foot-relative local frame (InsoleGeometry.to_local) before comparing;
    # reused verbatim here rather than re-deriving it.
    marker_trc = read_trc(trial.marker_trc)
    framework_trc = read_trc(trial.effective_opencap_trc())
    window_marker_local = InsoleGeometry.to_local(marker_trc, window_time - shift, window_marker_cop, trial.side)
    window_framework_local = InsoleGeometry.to_local(framework_trc, window_time, framework_cop, trial.side)

    return {
        "window_time": window_time,
        "window_marker_force": window_marker_force,
        "window_marker_cop": window_marker_cop,
        "window_marker_local": window_marker_local,
        "framework_force": framework_force,
        "framework_cop": framework_cop,
        "window_framework_local": window_framework_local,
        "full_marker_time": full_marker_time,
        "full_marker_force": full_marker_force,
        "full_marker_cop": full_marker_cop,
    }


class App:
    def __init__(self, root):
        self.root = root
        root.title("Force/CoP browser -- marker vs. framework (2D, fast)")
        root.geometry("1500x880")

        self.discovery = TrialDiscovery(DEFAULT)
        self.activity = activity_for(ACTIVITY_NAME, DEFAULT)
        self.trials_by_name = {}
        self.trial_names = []

        left = ttk.Frame(root, padding=6)
        left.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(left, text="SLJ trials (all subjects)", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.listing = tk.Listbox(left, width=28, height=40, exportselection=False)
        self.listing.pack(fill=tk.Y, expand=True, pady=5)
        self.listing.bind("<<ListboxSelect>>", self._on_select)
        self._populate()

        self.status = ttk.Label(left, text="Select a trial", wraplength=220)
        self.status.pack(fill=tk.X, pady=8)
        buttons = ttk.Frame(left)
        buttons.pack(fill=tk.X, pady=4)
        ttk.Button(buttons, text="Open marker in OpenSim", command=lambda: self._native("marker")).pack(fill=tk.X, pady=2)
        ttk.Button(buttons, text="Open framework in OpenSim", command=lambda: self._native("framework")).pack(fill=tk.X, pady=2)

        main = ttk.Frame(root)
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.figure = Figure(figsize=(11, 7.5))
        grid = self.figure.add_gridspec(2, 3, hspace=.4, wspace=.35)
        self.force_axes = [self.figure.add_subplot(grid[0, i]) for i in range(3)]
        self.cop_axes = [self.figure.add_subplot(grid[1, i]) for i in range(2)]
        self.cop_path_axis = self.figure.add_subplot(grid[1, 2])
        self.canvas = FigureCanvasTkAgg(self.figure, master=main)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.current = None

    def _populate(self):
        for subject_id in SUBJECT_IDS:
            subject = Subject(subject_id, DEFAULT)
            for trial in self.discovery.trials(subject, self.activity, "all", for_analysis=True):
                self.listing.insert(tk.END, trial.name)
                self.trial_names.append(trial.name)
                self.trials_by_name[trial.name] = trial

    def _on_select(self, _event=None):
        if not self.listing.curselection():
            return
        name = self.trial_names[self.listing.curselection()[0]]
        trial = self.trials_by_name[name]
        try:
            data = load_trial_forces(trial)
        except FileNotFoundError as error:
            self.status.configure(text=f"{name}\n\nNot available:\n{error}")
            self.current = None
            return
        self.current = (trial, data)
        self.status.configure(text=f"{name}\n{trial.condition} | stance side: {trial.side}\nLoaded")
        self._plot()

    def _plot(self):
        trial, d = self.current
        window_start, window_end = d["window_time"][0], d["window_time"][-1]
        labels = ("Fx (AP) GRF (N)", "Fy (vertical) GRF (N)", "Fz (ML) GRF (N)")
        for axis, label, index in zip(self.force_axes, labels, range(3)):
            axis.clear()
            axis.plot(d["full_marker_time"], d["full_marker_force"][:, index], color="0.7", linewidth=1, label="Marker (full trial)")
            axis.axvspan(window_start, window_end, color="tab:blue", alpha=0.08)
            axis.plot(d["window_time"], d["window_marker_force"][:, index], "k", label="Marker (window)")
            axis.plot(d["window_time"], d["framework_force"][:, index], "tab:red", label="Framework (window)")
            axis.set_title(label)
            axis.set_xlabel("Time (s)")
            axis.grid(True, alpha=0.3)
        self.force_axes[0].legend(fontsize=7, loc="upper left")

        # Foot-local frame (AP, ML relative to the foot) -- NOT raw
        # ground-frame CoP, which sits at two different absolute origins
        # for marker vs. framework and is not directly comparable (see
        # load_trial_forces()'s docstring note).
        cop_labels = ("CoP AP, foot-local (m)", "CoP ML, foot-local (m)")
        for axis, label, index in zip(self.cop_axes, cop_labels, (0, 1)):
            axis.clear()
            axis.axvspan(window_start, window_end, color="tab:blue", alpha=0.08)
            axis.plot(d["window_time"], d["window_marker_local"][:, index], "k", label="Marker")
            axis.plot(d["window_time"], d["window_framework_local"][:, index], "tab:red", label="Framework")
            axis.set_title(label)
            axis.set_xlabel("Time (s)")
            axis.grid(True, alpha=0.3)
        self.cop_axes[0].legend(fontsize=7)

        self.cop_path_axis.clear()
        self.cop_path_axis.plot(d["window_marker_local"][:, 0], d["window_marker_local"][:, 1], "k.-", label="Marker", markersize=3)
        self.cop_path_axis.plot(d["window_framework_local"][:, 0], d["window_framework_local"][:, 1], "tab:red", marker=".", linestyle="-", markersize=3, label="Framework")
        self.cop_path_axis.set_title("CoP path, foot-local frame (within analysis window)")
        self.cop_path_axis.set_xlabel("AP (m)")
        self.cop_path_axis.set_ylabel("ML (m)")
        self.cop_path_axis.axis("equal")
        self.cop_path_axis.grid(True, alpha=0.3)
        self.cop_path_axis.legend(fontsize=7)

        self.figure.suptitle(
            f"{trial.name}  (stance side: {trial.side})  --  shaded band = the +/-0.75s analysis window "
            "(framework/insole data only ever exists inside it; gray = marker's fuller recording, for context)",
            fontsize=9,
        )
        self.figure.tight_layout(rect=(0, 0, 1, 0.96))
        self.canvas.draw_idle()

    def _native(self, branch):
        if self.current is None:
            return
        trial, _data = self.current
        import subprocess
        subprocess.Popen([sys.executable, str(Path(CODE_ROOT / "expt_temp" / "visual_validation_gui" / "validation_gui.py")),
                           "--native", branch, "--trial", trial.name])


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
