# RTS Assessment -- SLJ Pipeline

Single Leg Jump (SLJ) vertical slice: raw C3D/insole/OpenCap data in, a
framework-vs-marker LSI comparison and a framework-vs-isokinetic correlation
out. See `../CLAUDE.md` for the full architecture writeup; this file is the
practical "how do I run it" reference.

## Setup

The `opensimPy/` venv (sibling to `data/`, `resources/`, `results/`) already
has everything installed: OpenSim 4.6, numpy, pandas, scipy, matplotlib,
openpyxl, pytest, requests.

```powershell
.\opensimPy\Scripts\Activate.ps1
```

## Commands

All commands run from the `code/` folder (or use `python code\cli.py ...`
from the repo root).

```powershell
# Pull one subject's OpenCap session(s) via the API (needs a credentials
# file, default ~/Documents/opencap-processing/.env with API_TOKEN=...).
python cli.py download --subject AB03

# See what data is usable today, without running anything expensive.
python cli.py preflight --subject AB08

# Ground-truth pipeline: C3D -> TRC/MOT -> scale -> IK -> ID.
python cli.py marker --subject AB08 --condition all
# ... or, if PROCESSED/MOCAP already has the converted files:
python cli.py marker --subject AB08 --condition all --skip-c3d

# Fit this subject's insole calibration (per-side force gain + rigid COP offset).
python cli.py calibrate --subject AB08

# Framework pipeline: OpenCap + insole -> bilateral GRF/COP -> ID, then LSI.
python cli.py framework --subject AB08 --condition all

# Rebuild the LSI comparison without reprocessing anything.
python cli.py analyze --subject AB08

# Join every subject's SLJ LSI against the isokinetic gold standard.
python cli.py correlate
```

Run the test suite:

```powershell
python -m pytest tests
```

## New subject checklist

1. Add a row to `data/subject_metadata.csv` (`SubjectID, Mass_kg, Height_m, ExoLeg, ShoeSize`).
2. Add `data/<ID>/RAW/MOCAP/*.c3d` and `data/<ID>/RAW/INSOLE/*.txt`.
3. Add their OpenCap session row(s) to `data/opencap_sessions.csv` (`SessionID` at minimum).
4. Download their OpenCap data: see `opencap_admin/download.py` (`OpenCapDownloader`) --
   needs an OpenCap API credentials file (default `~/Documents/opencap-processing/.env`
   with `API_TOKEN=...`).
5. Record any known-bad trials in `data/trial_status.csv`.
6. Add `subject_rules/<ID>.json` only when a documented exception is needed
   (a fixed calibration override, a landing-ordinal correction, an OpenCap
   filename alias -- see `subject_rules/default.json` for every field and
   `subject_rules/AB08.json`/`NTS01.json` for real examples).
7. Run `preflight` first, then `marker`, `calibrate`, `framework` in that order.

## Known real-world naming quirk

AB03's OpenCap trial files are named with a typo (`ABO3_...` -- letter O,
not zero) instead of `AB03_...`. This needs an `opencap_aliases` entry per
trial in `subject_rules/AB03.json` (see the mechanism already used for
NTS01/AB08 in their own rule files) before AB03's framework pipeline will
find its OpenCap TRC/IK files.

## What's not built yet

- The interactive TRC-repair GUI (`trc_repair.py` ports the automatic
  filter + matched-validation math from the in-progress prototype at
  `G:\Shared drives\SuhrudLib\PILOT_OOP_BASE\temp\AB08_SLJ_moment_investigation\trc_repair_gui\`,
  but not its manual per-marker review GUI).
- A GRF-accuracy investigation (sweeping `insole_vertical_weight`/
  `com_vertical_weight` and reporting agreement against ground truth) --
  see `resources/literature_notes.md` for why this matters.
- Triple hop / skater hop / lateral hop / agility activities (add a new
  `Activity` subclass in `activities.py` following `SingleLegJump`).
