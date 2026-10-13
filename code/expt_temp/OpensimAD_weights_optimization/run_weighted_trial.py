"""run_weighted_trial.py -- like ../opensimad_gain_check/run_one_trial.py,
but accepts an objective-weight override (JSON) applied to
`settings['weights']` after `processInputsOpenSimAD()` returns and before
`run_tracking()` -- no edits to the vendored OpenSimAD repo needed, since
`settings` is a plain dict `processInputsOpenSimAD` hands back and
`run_tracking` reads `settings['weights'][...]` directly (same pattern
already used for `settings['maxIter']`).

Must run under the `opencap-processing-grf` conda env (different Python/
CasADi/IPOPT install from this project's own opensimPy venv) -- invoked
via subprocess, never imported into this project's own process.

Usage (conda env's own python.exe):
    python run_weighted_trial.py <session_folder> <trial_name> \
        <window_start> <window_end> <case_name> <weights_overrides_json>

weights_overrides_json: a JSON object string, e.g.
    '{"accelerationTrackingTerm": 500}'
merged into the 'jumping' preset's own settings['weights'] (baseline:
positionTrackingTerm=100, velocityTrackingTerm=10,
accelerationTrackingTerm=50, activationTerm=1, plus the small excitation/
derivative terms -- see settingsOpenSimAD.py's 'jumping' setup) -- only
the keys present in the override are changed, everything else keeps the
preset's own default.
"""
import json
import os
import sys
import time
import traceback

OPENSIMAD_REPO = r"C:\Users\sjogleka\Documents\opencap-processing-grf-jumptest"


def main():
    session_folder, trial_name, window_start, window_end, case, weights_json = sys.argv[1:7]
    time_window = [float(window_start), float(window_end)]
    weight_overrides = json.loads(weights_json)

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

        print(f"Baseline weights: {settings['weights']}", flush=True)
        settings["weights"].update(weight_overrides)
        print(f"Weight overrides applied: {weight_overrides}", flush=True)
        print(f"Effective weights: {settings['weights']}", flush=True)

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
    finally:
        elapsed = time.time() - start
        print(f"STATUS: {status}", flush=True)
        print(f"ELAPSED_SECONDS: {elapsed:.1f}", flush=True)


if __name__ == "__main__":
    main()
