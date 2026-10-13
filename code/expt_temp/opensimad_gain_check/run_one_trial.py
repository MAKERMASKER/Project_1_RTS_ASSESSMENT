"""run_one_trial.py -- parametrized version of the external OpenSimAD repo's
own run_jump_ad.py (see README.md for exactly where that pattern came
from). Must run under the `opencap-processing-grf` conda env (different
Python/CasADi/IPOPT install from this project's own opensimPy venv) --
invoked via subprocess by launch_pilot.py, never imported into this
project's own process.

Usage (conda env's own python.exe):
    python run_one_trial.py <session_folder> <trial_name> <window_start> <window_end> <case_name> [max_iter]

max_iter (optional, default 5000, the tool's own default): raise this to
retry a trial that previously hit IPOPT's iteration cap without
converging -- same model/window, just a bigger iteration budget.
"""
import os
import sys
import time
import traceback

OPENSIMAD_REPO = r"C:\Users\sjogleka\Documents\opencap-processing-grf-jumptest"


def main():
    session_folder, trial_name, window_start, window_end, case = sys.argv[1:6]
    max_iter = int(sys.argv[6]) if len(sys.argv) > 6 else None
    time_window = [float(window_start), float(window_end)]

    base_dir = OPENSIMAD_REPO
    sys.path.append(base_dir)
    sys.path.append(os.path.join(base_dir, "UtilsDynamicSimulations", "OpenSimAD"))
    data_folder = os.path.join(base_dir, "Data")

    import matplotlib
    matplotlib.use("Agg")
    from utilsOpenSimAD import processInputsOpenSimAD, plotResultsOpenSimAD
    from mainOpenSimAD import run_tracking

    start = time.time()
    status = "unknown"
    try:
        print(f"Step 1: processInputsOpenSimAD ({trial_name}, window={time_window})...", flush=True)
        settings = processInputsOpenSimAD(
            base_dir, data_folder, session_folder, trial_name, "jumping",
            time_window=time_window, treadmill_speed=0,
        )
        print("processInputsOpenSimAD OK", flush=True)
        if max_iter is not None:
            settings["maxIter"] = max_iter
            print(f"maxIter overridden to {max_iter}", flush=True)

        print("Step 2: run_tracking (IPOPT trajectory optimization solve)...", flush=True)
        run_tracking(base_dir, data_folder, session_folder, settings, case=case)
        print("run_tracking OK", flush=True)
        status = "converged"

        print("Step 3: plotResultsOpenSimAD...", flush=True)
        plotResultsOpenSimAD(data_folder, session_folder, trial_name, settings, [case])
        print("ALL DONE", flush=True)
    except Exception as error:
        status = f"failed: {type(error).__name__}: {error}"
        print("FAILED:", type(error).__name__, error, flush=True)
        traceback.print_exc()
        # IPOPT hitting its iteration cap without an exception (as in the
        # reference project's own non-convergent run) won't land here --
        # that shows up as "converged" above but should be checked against
        # the solver log for Maximum_Iterations_Exceeded before trusting it.
    finally:
        elapsed = time.time() - start
        print(f"STATUS: {status}", flush=True)
        print(f"ELAPSED_SECONDS: {elapsed:.1f}", flush=True)


if __name__ == "__main__":
    main()
