"""OpenCapDownloader: pull one subject's OpenCap session outputs via its REST API.

Ported from resources/_old_reference_code/opencap_admin/download.py. Needed
for AB03/AB05/AB07: opencap_sessions.csv already has their SessionIDs (their
`Folder` column is blank because the data hasn't been pulled down yet -- see
CLAUDE.md's Domain facts).

Requires an OpenCap API credentials file (default
~/Documents/opencap-processing/.env) with API_TOKEN= (and optionally
API_URL=). This module only builds the download logic; running it against
the real API requires that credentials file to exist -- nothing here can
supply or validate the token itself.
"""
from pathlib import Path
from urllib.parse import urlparse
import re
import shutil
import time

import pandas as pd
import requests

from config import DEFAULT

DEFAULT_CREDENTIALS_PATH = Path.home() / "Documents" / "opencap-processing" / ".env"
DEFAULT_API_URL = "https://api.opencap.ai/"
NON_TRIAL_NAMES = ("neutral", "calibration")
RETRYABLE_ERRORS = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_S = 5.0


def _with_retries(description: str, call):
    """The OpenCap API occasionally drops a connection mid-request (observed:
    'Remote end closed connection without response') with no server-side
    error -- a plain retry succeeds. Retry a handful of times with a short
    backoff before giving up for real."""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return call()
        except RETRYABLE_ERRORS as error:
            if attempt == RETRY_ATTEMPTS:
                raise
            print(f"  {description}: attempt {attempt} failed ({error}); retrying...", flush=True)
            time.sleep(RETRY_BACKOFF_S * attempt)


class OpenCapDownloader:
    def __init__(self, config=DEFAULT, credentials_path: Path = None):
        self.config = config
        self.credentials_path = Path(credentials_path) if credentials_path else DEFAULT_CREDENTIALS_PATH

    def credentials(self) -> dict:
        path = self.credentials_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing OpenCap credentials file: {path}")
        text = path.read_text()
        values = {}
        for name in ("API_TOKEN", "API_URL"):
            match = re.search(rf'^\s*{name}\s*=\s*["\']?([^"\'\r\n]+)', text, re.M)
            values[name] = match.group(1).strip() if match else ""
        if not values["API_TOKEN"]:
            raise ValueError(f"API_TOKEN is missing from {path}")
        values["API_URL"] = (values["API_URL"] or DEFAULT_API_URL).rstrip("/") + "/"
        return values

    def sessions_for(self, subject) -> pd.DataFrame:
        manifest = pd.read_csv(self.config.session_file, dtype=str)
        selected = manifest[manifest["SubjectID"].astype(str).str.upper() == subject.id]
        if selected.empty:
            raise ValueError(f"No OpenCap sessions listed for {subject.id} in {self.config.session_file}")
        return selected

    def download_subject(self, subject) -> list:
        """Download every session listed for this subject; mirror RAW -> PROCESSED.

        Returns the list of OpenCapData_<id> folder names downloaded.
        """
        auth = self.credentials()
        api = requests.Session()
        api.headers["Authorization"] = f"Token {auth['API_TOKEN']}"

        folders = []
        failed = []
        for _, row in self.sessions_for(subject).iterrows():
            session_id = row["SessionID"]
            try:
                folders.append(self._download_session(subject, session_id, api, auth["API_URL"]))
            except RETRYABLE_ERRORS as error:
                # One session's transient network failure (even after
                # retries) must not lose progress already made on this
                # subject's other sessions in the same call.
                print(f"  session {session_id}: giving up after retries ({error})", flush=True)
                failed.append(session_id)

        self._mirror_to_processed(subject, folders)
        self._record_folders_in_manifest(subject, folders)
        if failed:
            print(f"{subject.id}: {len(failed)} session(s) failed, re-run download_subject to retry: {failed}")
        return folders

    def _record_folders_in_manifest(self, subject, folders: list):
        """Write the deterministic OpenCapData_<id> folder name back into
        opencap_sessions.csv's Folder column, only for sessions that actually
        downloaded -- for auditability. Discovery itself doesn't require this
        (it falls back to SessionID), but a human reading the manifest should
        be able to see what's downloaded without checking the filesystem."""
        if not folders:
            return
        path = self.config.session_file
        manifest = pd.read_csv(path, dtype=str)
        rows = (manifest["SubjectID"].astype(str).str.upper() == subject.id) & (
            ("OpenCapData_" + manifest["SessionID"].astype(str)).isin(folders)
        )
        manifest.loc[rows, "Folder"] = "OpenCapData_" + manifest.loc[rows, "SessionID"].astype(str)
        manifest.to_csv(path, index=False)

    def _download_session(self, subject, session_id: str, api: requests.Session, api_url: str) -> str:
        folder = f"OpenCapData_{session_id}"
        destination = subject.data_folder / "RAW" / "OPENCAP" / folder
        destination.mkdir(parents=True, exist_ok=True)
        print(f"OpenCap session {session_id}", flush=True)

        response = _with_retries(
            f"session {session_id} info",
            lambda: api.get(f"{api_url}sessions/{session_id}/", timeout=120),
        )
        response.raise_for_status()
        trials = response.json()["trials"]
        neutral = next((trial for trial in reversed(trials) if trial["name"].lower() == "neutral"), None)
        if neutral is None:
            raise ValueError(f"Session {session_id} has no neutral trial")

        metadata = self._result_by_tag(neutral, "session_metadata")
        if metadata:
            self._download_media(metadata["media"], destination / "sessionMetadata.yaml")
        model = self._result_by_tag(neutral, "opensim_model")
        if model:
            name = self._media_name(model["media"], "LaiUhlrich2022_scaled.osim")
            self._download_media(model["media"], destination / "OpenSimData" / "Model" / name)

        for trial in trials:
            if trial["name"].lower() in NON_TRIAL_NAMES:
                continue
            outputs = (
                ("marker_data", destination / "MarkerData" / f"{trial['name']}.trc"),
                ("ik_results", destination / "OpenSimData" / "Kinematics" / f"{trial['name']}.mot"),
                ("main_settings", destination / "MarkerData" / "Settings" / f"settings_{trial['name']}.yaml"),
            )
            for tag, path in outputs:
                result = self._result_by_tag(trial, tag)
                if result:
                    self._download_media(result["media"], path)
        return folder

    def _mirror_to_processed(self, subject, folders: list):
        raw = subject.data_folder / "RAW" / "OPENCAP"
        processed = subject.data_folder / "PROCESSED" / "OPENCAP"
        for folder in folders:
            target = processed / folder
            if not target.is_dir():
                shutil.copytree(raw / folder, target)

    @staticmethod
    def _result_by_tag(trial: dict, tag: str):
        return next((item for item in trial.get("results", []) if item.get("tag") == tag), None)

    @staticmethod
    def _media_name(url: str, fallback: str) -> str:
        name = Path(urlparse(url).path).name
        return name.rsplit("-", 1)[-1] if name else fallback

    @staticmethod
    def _download_media(url: str, destination: Path):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            return
        print(" ", destination, flush=True)
        response = _with_retries(str(destination), lambda: requests.get(url, timeout=180))
        response.raise_for_status()
        destination.write_bytes(response.content)
