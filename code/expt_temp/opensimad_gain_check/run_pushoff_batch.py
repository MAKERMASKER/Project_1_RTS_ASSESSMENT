"""run_pushoff_batch.py -- run the push-off (jump-initiation) window
through OpenSimAD for every planned trial (trial_plan_pushoff.json),
sequentially (not parallel -- avoid CPU/memory contention between heavy
IPOPT solves), skipping any trial whose result already exists.

Must run from a shell with the opencap-processing-grf conda env's tools on
PATH -- invokes conda run itself per trial, same as the manual pilot runs.

Run (from code/, in the opensimPy environment or plain cmd/powershell):
    python expt_temp/opensimad_gain_check/run_pushoff_batch.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

PLAN_PATH = Path(__file__).resolve().parent / "trial_plan_pushoff.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RUNNER = Path(__file__).resolve().parent / "run_one_trial.py"
CONDA = r"C:\Users\sjogleka\AppData\Local\anaconda3\condabin\conda.bat"
OPENSIMAD_DATA_ROOT = Path(r"C:\Users\sjogleka\Documents\opencap-processing-grf-jumptest\Data")


def main():
    plan = json.loads(PLAN_PATH.read_text())
    for entry in plan:
        case = f"{entry['subject'].lower()}_{entry['side'].lower()}_pushoff"
        out_dir = RESULTS_DIR / f"{entry['trial_name']}_pushoff"
        grf_path = out_dir / "GRF.csv"
        if grf_path.is_file():
            print(f"{entry['trial_name']}: already have GRF.csv -- skipped")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "run.log"
        window = entry["time_window"]
        print(f"{entry['trial_name']}: launching push-off window {window} (session {entry['session_folder']})")

        start = time.time()
        with log_path.open("w") as log_file:
            result = subprocess.run(
                [CONDA, "run", "-n", "opencap-processing-grf", "--no-capture-output", "python",
                 str(RUNNER), entry["session_folder"], entry["opencap_trial_name"],
                 str(window[0]), str(window[1]), case],
                stdout=log_file, stderr=subprocess.STDOUT,
            )
        elapsed = time.time() - start
        print(f"  exit_code={result.returncode}  elapsed={elapsed:.0f}s")

        source_grf = (
            OPENSIMAD_DATA_ROOT / entry["session_folder"] / "OpenSimData" / "Dynamics"
            / entry["opencap_trial_name"] / f"Results_{case}" / "GRF.csv"
        )
        if source_grf.is_file():
            grf_path.write_bytes(source_grf.read_bytes())
            print(f"  copied -> {grf_path}")
        else:
            print(f"  WARNING: no GRF.csv produced at {source_grf} -- check {log_path}")


if __name__ == "__main__":
    main()
