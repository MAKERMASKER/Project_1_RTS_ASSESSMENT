"""prepare_session_data.py -- mirror one subject's OpenCap session folder
(MarkerData/OpenSimData/sessionMetadata.yaml) from this project's own
data/<SUBJECT>/RAW/OPENCAP/<session>/ into the external OpenSimAD repo's
Data/<session>/, if not already there. Pure file copy -- no compute, no
OpenSimAD invocation.

Run (from code/, in the opensimPy environment):
    python expt_temp/opensimad_gain_check/prepare_session_data.py
"""
import json
import shutil
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from config import DEFAULT  # noqa: E402
from subject import Subject  # noqa: E402

PLAN_PATH = Path(__file__).resolve().parent / "trial_plan.json"
OPENSIMAD_DATA_ROOT = Path(r"C:\Users\sjogleka\Documents\opencap-processing-grf-jumptest\Data")


def mirror_session(subject_id: str, session_folder: str):
    destination = OPENSIMAD_DATA_ROOT / session_folder
    if destination.is_dir():
        print(f"  {session_folder}: already present -- skipped")
        return
    subject = Subject(subject_id, DEFAULT)
    source = subject.data_folder / "RAW" / "OPENCAP" / session_folder
    if not source.is_dir():
        raise FileNotFoundError(f"No RAW OpenCap session at {source}")
    shutil.copytree(source, destination)
    print(f"  {session_folder}: copied {source} -> {destination}")


def main():
    plan = json.loads(PLAN_PATH.read_text())
    seen = set()
    for entry in plan:
        key = (entry["subject"], entry["session_folder"])
        if key in seen:
            continue
        seen.add(key)
        print(f"{entry['subject']}: {entry['session_folder']}")
        mirror_session(entry["subject"], entry["session_folder"])


if __name__ == "__main__":
    main()
