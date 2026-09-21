"""Synchronized marker/framework visual validation GUI.

Ported from the reference prototype at
G:\\Shared drives\\SuhrudLib\\PILOT_OOP_BASE_PY\\NONOOP\\temp\\AB08_independent_planar_handcheck\\visual_validation_gui\\validation_gui.py
-- same architecture (matplotlib-rendered synchronized marker/framework
3D+2D dashboard, native-OpenSim "Open in OpenSim" buttons as a secondary,
separate view), not a redesign. What changed vs. the reference, and why:

- All old-repo-specific imports/paths (`PYROOT`, `COMMON`,
  `corrected_validation_force_mapping`, `data_io`, `insole`, `opensim_tools`,
  `study`) replaced with this project's own modules (`config`, `subject`,
  `discovery`, `activities`, `io_tables`, `opensim_env`, `insole_geometry`)
  and this project's actual `results/` layout -- the reference pointed at a
  different repo entirely, which doesn't exist here.
- `map_forceplate_to_opencap` (an old-repo helper, not present in this
  project) is replaced by the marker force-plate signal already computed a
  few lines above it (`marker_force`) -- it served the same role in the
  Fx/Fy/Fz comparison panel.
- The marker-vs-framework time alignment ("shift") is computed the same way
  this project's own `outcomes.py.analyze_trial()` already does it
  (framework's own vertical-force peak vs. the force-plate's own vertical-
  force peak) instead of reading a `summary.csv`'s `AlignmentShift_s` column
  that doesn't exist in this project.
- Trial list covers all 5 subjects (AB03/AB05/AB07/AB08/NTS01), not just
  AB08 -- everything else about the list/selection/caching is unchanged.
- `native_play()`'s OpenSim-visualizer launch adds the Geometry search path
  and works around a real environment quirk found while building this
  (two OpenSim installs on this machine; PATH resolves `simbody-
  visualizer.exe` to an older, version-mismatched one unless corrected).

Run from the opensimPy environment:
    python validation_gui.py

The left list is generated from `TrialDiscovery(...).trials(..., for_analysis=True)`
across all 5 subjects, so it contains only valid, analyzable SLJ trials.
Selecting a trial shows synchronized marker IK + force-plate and framework
IK + framework GRF panels. The bottom row shows foot-frame CoP and
common-frame Fx/Fy/Fz. Drag the slider, press Space, or use the arrow keys.
Full model-mesh playback can be opened in native OpenSim windows via the
buttons (each a separate, non-synced native visualizer window -- a known,
accepted limitation, same as in the reference).

The first load of a trial builds a local cache; subsequent viewing is
immediate. The GUI is read-only and does not change marker, framework, IK,
GRF, or ID results.
"""
import os
import sys
import subprocess
import threading
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import pandas as pd

_OPENSIM_46_BIN = r"C:\OpenSim 4.6\bin"
if Path(_OPENSIM_46_BIN).is_dir():
    os.environ["PATH"] = _OPENSIM_46_BIN + os.pathsep + os.environ.get("PATH", "")
GEOMETRY = Path(r"C:\OpenSim 4.6\Geometry")

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
CODE_ROOT = HERE.resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from activities import activity_for
from config import DEFAULT
from discovery import TrialDiscovery
from insole_geometry import InsoleGeometry
from io_tables import read_storage, read_trc
from opensim_env import opensim_module
from subject import Subject

SUBJECT_IDS = ("AB03", "AB05", "AB07", "AB08", "NTS01")
ACTIVITY_NAME = "single_leg_jump"

BODIES = ("pelvis", "torso", "femur_r", "tibia_r", "talus_r", "calcn_r", "toes_r",
          "femur_l", "tibia_l", "talus_l", "calcn_l", "toes_l", "humerus_r", "ulna_r",
          "radius_r", "hand_r", "humerus_l", "ulna_l", "radius_l", "hand_l")
EDGES = (("pelvis", "torso"), ("pelvis", "femur_r"), ("femur_r", "tibia_r"), ("tibia_r", "talus_r"),
         ("talus_r", "calcn_r"), ("calcn_r", "toes_r"), ("pelvis", "femur_l"), ("femur_l", "tibia_l"),
         ("tibia_l", "talus_l"), ("talus_l", "calcn_l"), ("calcn_l", "toes_l"), ("torso", "humerus_r"),
         ("humerus_r", "ulna_r"), ("ulna_r", "radius_r"), ("radius_r", "hand_r"), ("torso", "humerus_l"),
         ("humerus_l", "ulna_l"), ("ulna_l", "radius_l"), ("radius_l", "hand_l"))


def model_path_for(subject_id, condition):
    return DEFAULT.results_root / subject_id / "marker_based" / "models" / f"{subject_id}_{condition}_LaiUhlrich2022.osim"


def marker_ik_path_for(trial):
    return DEFAULT.results_root / trial.subject / "marker_based" / trial.condition / "ik" / f"{trial.name}_ik.mot"


def interpolate_xyz(trc, time):
    xyz = trc["xyz"].astype(float, copy=True)
    if np.nanmedian(np.abs(xyz)) > 2:
        xyz /= 1000
    out = np.empty((len(time), xyz.shape[1], 3))
    for marker in range(xyz.shape[1]):
        for axis in range(3):
            out[:, marker, axis] = np.interp(time, trc["time"], xyz[:, marker, axis], left=np.nan, right=np.nan)
    return out


def model_motion(model_file, ik, time):
    osim = opensim_module()
    osim.ModelVisualizer.addDirToGeometrySearchPaths(str(GEOMETRY))
    model = osim.Model(str(model_file))
    state = model.initSystem()
    coordinates = model.getCoordinateSet()
    names = [name for name in ik.columns if name != "time" and coordinates.contains(name)]
    values = ik[names].to_numpy(float, copy=True)
    rotational = [i for i, n in enumerate(names) if n not in ("pelvis_tx", "pelvis_ty", "pelvis_tz")]
    if rotational and np.nanmax(np.abs(values[:, rotational])) > 6.5:
        values[:, rotational] = np.deg2rad(values[:, rotational])
    values = np.column_stack([np.interp(time, ik["time"], values[:, i], left=np.nan, right=np.nan) for i in range(len(names))])
    body_names = [name for name in BODIES if model.getBodySet().contains(name)]
    marker_names = [model.getMarkerSet().get(i).getName() for i in range(model.getMarkerSet().getSize())]
    bodies = np.zeros((len(time), len(body_names), 3))
    markers = np.zeros((len(time), len(marker_names), 3))
    for row in range(len(time)):
        for column, name in enumerate(names):
            value = values[row, column]
            if np.isfinite(value):
                coordinates.get(name).setValue(state, float(value))
        model.realizePosition(state)
        for i, name in enumerate(body_names):
            position = model.getBodySet().get(name).getPositionInGround(state)
            bodies[row, i] = [position.get(j) for j in range(3)]
        for i in range(len(marker_names)):
            location = model.getMarkerSet().get(i).getLocationInGround(state)
            markers[row, i] = [location.get(j) for j in range(3)]
    return body_names, marker_names, bodies, markers


def load_trial(trial):
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{trial.name}.npz"
    if cache.is_file():
        return dict(np.load(cache, allow_pickle=True))

    package = np.load(trial.output / "trial_data.npz", allow_pickle=True)
    time = package["time"].astype(float)
    side = "right" if trial.side == "R" else "left"
    model = model_path_for(trial.subject, trial.condition)

    marker_ik = read_storage(marker_ik_path_for(trial))
    framework_ik = read_storage(trial.effective_opencap_ik())

    # Marker-based mocap is its own, independent recording session -- align
    # it to the framework's timeline the same way outcomes.py.analyze_trial()
    # already does it (framework's own vertical-force peak vs. the
    # force-plate's own vertical-force peak), not via a summary.csv column
    # from a different project that doesn't exist here.
    force_plate = read_storage(trial.force_mot)
    plate = 1 if trial.side == "R" else 2
    marker_vertical = np.abs(force_plate[f"ground_force_{plate}_vy"].to_numpy(float))
    shift = float(time[np.argmax(package["force"])] - force_plate["time"].iloc[np.argmax(marker_vertical)])

    mbn, mmn, marker_bodies, marker_model_markers = model_motion(model, marker_ik, time - shift)
    fbn, fmn, framework_bodies, framework_model_markers = model_motion(model, framework_ik, time)

    marker_trc = read_trc(trial.marker_trc)
    framework_trc = read_trc(trial.effective_opencap_trc())
    marker_observed = interpolate_xyz(marker_trc, time - shift)
    framework_observed = interpolate_xyz(framework_trc, time)

    marker_force = np.column_stack([
        np.interp(time - shift, force_plate["time"], force_plate[f"ground_force_{plate}_v{axis}"], left=np.nan, right=np.nan)
        for axis in "xyz"
    ])
    marker_cop = np.column_stack([
        np.interp(time - shift, force_plate["time"], force_plate[f"ground_force_{plate}_p{axis}"], left=np.nan, right=np.nan)
        for axis in "xyz"
    ])
    if np.nanmedian(np.abs(marker_cop)) > 2:
        marker_cop /= 1000
    marker_contact = np.abs(marker_force[:, 1]) >= 25
    marker_cop[~marker_contact] = np.nan
    marker_force[~marker_contact] = 0

    framework_force = package[f"{side}_grf_N"].astype(float)
    framework_cop = package[f"{side}_cop_m"].astype(float)
    framework_contact = package[f"{side}_contact"].astype(bool)
    framework_cop[~framework_contact] = np.nan
    framework_force[~framework_contact] = 0

    marker_local = InsoleGeometry.to_local(marker_trc, time - shift, marker_cop, trial.side)
    framework_local = InsoleGeometry.to_local(framework_trc, time, framework_cop, trial.side)

    result = dict(
        time=time, shift=np.array(shift),
        marker_body_names=np.array(mbn), framework_body_names=np.array(fbn),
        marker_model_marker_names=np.array(mmn), framework_model_marker_names=np.array(fmn),
        marker_observed_names=np.array(marker_trc["marker_names"]), framework_observed_names=np.array(framework_trc["marker_names"]),
        marker_bodies=marker_bodies, framework_bodies=framework_bodies,
        marker_model_markers=marker_model_markers, framework_model_markers=framework_model_markers,
        marker_observed=marker_observed, framework_observed=framework_observed,
        marker_force=marker_force, marker_cop=marker_cop,
        framework_force=framework_force, framework_cop=framework_cop,
        marker_compare=marker_force, marker_local=marker_local, framework_local=framework_local,
    )
    np.savez_compressed(cache, **result)
    return result


def _find_trial(trial_name):
    subject_id = trial_name.split("_")[0]
    subject = Subject(subject_id, DEFAULT)
    activity = activity_for(ACTIVITY_NAME, DEFAULT)
    for trial in TrialDiscovery(DEFAULT).trials(subject, activity, "all", for_analysis=True):
        if trial.name == trial_name:
            return trial
    raise ValueError(f"Trial not found: {trial_name}")


def native_play(trial_name, branch):
    trial = _find_trial(trial_name)
    model_file = model_path_for(trial.subject, trial.condition)
    motion = marker_ik_path_for(trial) if branch == "marker" else trial.effective_opencap_ik()
    osim = opensim_module()
    osim.ModelVisualizer.addDirToGeometrySearchPaths(str(GEOMETRY))
    model = osim.Model(str(model_file))
    model.updDisplayHints().set_show_markers(True)
    table = osim.TimeSeriesTable(str(motion))
    osim.VisualizerUtilities.showMotion(model, table)


def run_gui():
    import tkinter as tk
    from tkinter import ttk, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure

    activity = activity_for(ACTIVITY_NAME, DEFAULT)
    discovery = TrialDiscovery(DEFAULT)
    trials = []
    for subject_id in SUBJECT_IDS:
        subject = Subject(subject_id, DEFAULT)
        trials.extend(discovery.trials(subject, activity, "all", for_analysis=True))

    root = tk.Tk()
    root.title("Marker / framework synchronized validation")
    root.geometry("1750x1000")
    left = ttk.Frame(root, padding=6)
    left.pack(side=tk.LEFT, fill=tk.Y)
    ttk.Label(left, text="Valid SLJ trials", font=("Segoe UI", 12, "bold")).pack(anchor="w")
    listing = tk.Listbox(left, width=28, height=36, exportselection=False)
    listing.pack(fill=tk.Y, expand=True, pady=5)
    for trial in trials:
        listing.insert(tk.END, trial.name)
    main = ttk.Frame(root)
    main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    fig = Figure(figsize=(15, 9))
    grid = fig.add_gridspec(2, 5, height_ratios=[2.2, 1], hspace=.3, wspace=.35)
    marker3d = fig.add_subplot(grid[0, 0:2], projection="3d")
    framework3d = fig.add_subplot(grid[0, 2:5], projection="3d")
    copax = fig.add_subplot(grid[1, 0:2])
    force_axes = [fig.add_subplot(grid[1, i]) for i in range(2, 5)]
    canvas = FigureCanvasTkAgg(fig, master=main)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    toolbar = NavigationToolbar2Tk(canvas, main)
    toolbar.update()
    controls = ttk.Frame(main, padding=5)
    controls.pack(fill=tk.X)
    play_button = ttk.Button(controls, text="Play")
    play_button.pack(side=tk.LEFT)
    slider = tk.Scale(controls, from_=0, to=1, orient=tk.HORIZONTAL, showvalue=False, resolution=1)
    slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
    time_label = ttk.Label(controls, text="Time: --")
    time_label.pack(side=tk.LEFT, padx=8)
    marker_native = ttk.Button(controls, text="Open marker in OpenSim")
    marker_native.pack(side=tk.LEFT, padx=4)
    framework_native = ttk.Button(controls, text="Open framework in OpenSim")
    framework_native.pack(side=tk.LEFT, padx=4)
    status = ttk.Label(left, text="Select a trial", wraplength=220)
    status.pack(fill=tk.X, pady=8)
    state = {"data": None, "trial": None, "playing": False, "token": 0, "artists": {}, "loading": False}

    def limits(axis, points):
        flat = points.reshape(-1, 3)
        valid = flat[np.isfinite(flat).all(1)]
        low = np.percentile(valid, 1, axis=0)
        high = np.percentile(valid, 99, axis=0)
        centre = (low + high) / 2
        span = max(np.max(high - low), .8) / 2
        axis.set_xlim(centre[0] - span, centre[0] + span)
        axis.set_ylim(centre[2] - span, centre[2] + span)
        axis.set_zlim(max(-.05, centre[1] - span), centre[1] + span)
        axis.set_xlabel("X")
        axis.set_ylabel("Z")
        axis.set_zlabel("Y")
        axis.set_box_aspect((1, 1, 1))

    def setup_plot():
        d = state["data"]
        trial = state["trial"]
        for axis, title in ((marker3d, "Marker IK + force plate"), (framework3d, "Framework IK + framework GRF")):
            axis.clear()
            axis.set_title(f"{trial.name}  (stance side: {trial.side})\n{title}")
        limits(marker3d, np.concatenate((d["marker_bodies"], d["marker_observed"]), axis=1))
        limits(framework3d, np.concatenate((d["framework_bodies"], d["framework_observed"]), axis=1))
        state["artists"] = {"lines": [], "quivers": []}
        for axis, prefix in ((marker3d, "marker"), (framework3d, "framework")):
            names = [str(x) for x in d[f"{prefix}_body_names"]]
            index = {name: i for i, name in enumerate(names)}
            for first, second in EDGES:
                if first in index and second in index:
                    line, = axis.plot([], [], [], color="0.25", lw=2)
                    state["artists"]["lines"].append((line, prefix, index[first], index[second]))
            predicted = axis.scatter([], [], [], s=13, c="tab:blue", label="Model markers")
            observed = axis.scatter([], [], [], s=15, c="tab:orange", marker="x", label="Observed markers")
            cop = axis.scatter([], [], [], s=55, c="magenta", label="CoP")
            state["artists"][f"{prefix}_predicted"] = predicted
            state["artists"][f"{prefix}_observed"] = observed
            state["artists"][f"{prefix}_cop"] = cop
            axis.legend(loc="upper right", fontsize=8)
        copax.clear()
        copax.plot(d["marker_local"][:, 0], d["marker_local"][:, 1], "k", label="Marker force plate")
        copax.plot(d["framework_local"][:, 0], d["framework_local"][:, 1], "tab:red", label="Framework insole")
        state["artists"]["marker_cop_point"] = copax.scatter([], [], c="k", s=45)
        state["artists"]["framework_cop_point"] = copax.scatter([], [], c="tab:red", s=45)
        copax.set(xlabel="Foot AP (m)", ylabel="Foot ML (m)", title="Foot-frame CoP")
        copax.axis("equal")
        copax.grid(alpha=.25)
        copax.legend(fontsize=8)
        for i, (axis, name) in enumerate(zip(force_axes, ("Fx", "Fy", "Fz"))):
            axis.clear()
            axis.plot(d["time"], d["marker_compare"][:, i], "k", label="Marker")
            axis.plot(d["time"], d["framework_force"][:, i], "tab:red", label="Framework")
            cursor = axis.axvline(d["time"][0], color="tab:blue", ls="--")
            state["artists"][f"cursor_{i}"] = cursor
            axis.set_title(name)
            axis.set_xlabel("Time (s)")
            axis.set_ylabel("N")
            axis.grid(alpha=.25)
            if i == 0:
                axis.legend(fontsize=8)
        slider.configure(to=len(d["time"]) - 1)
        slider.set(0)
        update_frame(0)
        canvas.draw_idle()

    def update_frame(value):
        if state["data"] is None:
            return
        d = state["data"]
        i = int(float(value))
        i = max(0, min(i, len(d["time"]) - 1))
        scale = .55 / max(np.nanmax(np.linalg.norm(d["marker_force"], axis=1)), np.nanmax(np.linalg.norm(d["framework_force"], axis=1)), 1)
        for line, prefix, a, b in state["artists"]["lines"]:
            points = d[f"{prefix}_bodies"][i, [a, b]]
            line.set_data_3d(points[:, 0], points[:, 2], points[:, 1])
        for prefix in ("marker", "framework"):
            predicted = d[f"{prefix}_model_markers"][i]
            observed = d[f"{prefix}_observed"][i]
            cop = d[f"{prefix}_cop"][i]
            state["artists"][f"{prefix}_predicted"]._offsets3d = (predicted[:, 0], predicted[:, 2], predicted[:, 1])
            state["artists"][f"{prefix}_observed"]._offsets3d = (observed[:, 0], observed[:, 2], observed[:, 1])
            state["artists"][f"{prefix}_cop"]._offsets3d = ([cop[0]], [cop[2]], [cop[1]]) if np.isfinite(cop).all() else ([], [], [])
        for quiver in state["artists"]["quivers"]:
            quiver.remove()
        state["artists"]["quivers"] = []
        for axis, prefix in ((marker3d, "marker"), (framework3d, "framework")):
            cop = d[f"{prefix}_cop"][i]
            force = d[f"{prefix}_force"][i]
            if np.isfinite(cop).all() and np.linalg.norm(force) > 0:
                q = axis.quiver(cop[0], cop[2], cop[1], force[0] * scale, force[2] * scale, force[1] * scale,
                                 color="red", linewidth=2, arrow_length_ratio=.12)
                state["artists"]["quivers"].append(q)
        ml = d["marker_local"][i]
        fl = d["framework_local"][i]
        state["artists"]["marker_cop_point"].set_offsets(ml[:2] if np.isfinite(ml).all() else np.empty((0, 2)))
        state["artists"]["framework_cop_point"].set_offsets(fl[:2] if np.isfinite(fl).all() else np.empty((0, 2)))
        for n in range(3):
            state["artists"][f"cursor_{n}"].set_xdata([d["time"][i], d["time"][i]])
        time_label.configure(text=f"Frame {i + 1}/{len(d['time'])} | {d['time'][i]:.3f} s")
        canvas.draw_idle()

    slider.configure(command=update_frame)

    def select(_=None):
        if not listing.curselection():
            return
        # A background thread doing per-frame OpenSim forward-kinematics is
        # CPU-bound, and Python's GIL means that still contends with the Tk
        # main loop for the interpreter -- clicking several trials in quick
        # succession before the first finishes piles up multiple such
        # threads, starving the UI ("Not Responding"). One load at a time,
        # with visible feedback, instead of silently queuing more of them.
        if state["loading"]:
            status.configure(text=f"Still loading {state.get('loading_name', '...')} -- please wait for it to finish.")
            return
        trial = trials[listing.curselection()[0]]
        state["token"] += 1
        token = state["token"]
        state["loading"] = True
        state["loading_name"] = trial.name
        listing.configure(state=tk.DISABLED)
        status.configure(text=f"Loading {trial.name}... first load builds a cache (can take a while)")

        def worker():
            try:
                data = load_trial(trial)
                error = None
            except Exception as exc:
                data = None
                error = exc

            def finish():
                state["loading"] = False
                listing.configure(state=tk.NORMAL)
                if token != state["token"]:
                    return
                if error:
                    status.configure(text=f"Failed: {error}")
                    messagebox.showerror("Load failed", str(error))
                    return
                state["trial"], state["data"] = trial, data
                status.configure(
                    text=f"{trial.name}\n{trial.condition} | stance side: {trial.side}\nLoaded and synchronized"
                )
                setup_plot()
            root.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    listing.bind("<<ListboxSelect>>", select)

    def toggle():
        state["playing"] = not state["playing"]
        play_button.configure(text="Pause" if state["playing"] else "Play")
        tick()

    def tick():
        if not state["playing"] or state["data"] is None:
            return
        value = int(slider.get()) + 1
        if value >= len(state["data"]["time"]):
            value = 0
        slider.set(value)
        root.after(35, tick)

    play_button.configure(command=toggle)

    def launch(branch):
        if state["trial"] is None:
            return
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--native", branch, "--trial", state["trial"].name])

    marker_native.configure(command=lambda: launch("marker"))
    framework_native.configure(command=lambda: launch("framework"))
    root.bind("<space>", lambda _: toggle())
    root.bind("<Right>", lambda _: slider.set(min(slider.get() + 1, slider.cget("to"))))
    root.bind("<Left>", lambda _: slider.set(max(slider.get() - 1, 0)))
    if trials:
        listing.selection_set(0)
        select()
    root.mainloop()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", choices=("marker", "framework"))
    parser.add_argument("--trial")
    args = parser.parse_args()
    if args.native:
        if not args.trial:
            parser.error("--trial is required with --native")
        native_play(args.trial, args.native)
    else:
        run_gui()


if __name__ == "__main__":
    main()
