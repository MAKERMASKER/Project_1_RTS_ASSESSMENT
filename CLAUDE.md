# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A from-scratch, object-oriented Python/OpenSim pipeline for an RTS (Return-to-Sport)
assessment study. It compares a markerless-mocap-plus-insole "framework" pipeline
against a marker-based-mocap-plus-force-plate "ground truth" pipeline, both run
through OpenSim, and ultimately correlates the resulting limb-symmetry index (LSI)
against a gold-standard isokinetic dynamometer LSI for the same subjects.

Current scope is **Single Leg Jump (SLJ) only** -- a full vertical slice from raw
data to an LSI number comparable against isokinetic LSI -- deliberately architected
so a second activity (triple hop, skater hop, ...) is a new class, not a rewrite.

There is no git repository here; this is a shared-drive working folder, not a repo.

## Repository layout

```
code/                    the actual codebase (see "Architecture" below)
data/                    real subject data -- see "Data layout" below
resources/
  _old_reference_code/   a prior, working pipeline for a structurally identical
                         but different study (exoskeleton-assist, not RTS) -- the
                         template this codebase ports algorithms from, rewritten
                         for clarity. Treat it as reference, not something to import.
  _opensim_reference/    generic OpenSim 4.6 API examples/tutorials, not project-specific
  literature_notes.md    standing literature-review artifact (insole/markerless GRF
                         accuracy, COP calibration, LSI/RTS clinical context) --
                         add findings here, don't let them live only in chat
opensimPy/               the Python 3.11 virtual environment (OpenSim 4.6 + deps)
results/                 pipeline outputs, one subfolder per subject (gitignore-style;
                         nothing here should be treated as source)
```

## Environment / commands

The venv at `opensimPy/` already has OpenSim 4.6, numpy, pandas, scipy, matplotlib,
openpyxl, and pytest installed.

Activate it (PowerShell, from repo root):
```powershell
.\opensimPy\Scripts\Activate.ps1
```

Or invoke the interpreter directly without activating (works from any shell,
including the Bash tool where the repo is mounted under `/g/Shared drives/...`):
```
"G:\Shared drives\SuhrudLib2\Projects\Project_1_RTS_ASSESSMENT\opensimPy\Scripts\python.exe" <script>.py
```

Install an additional package into this venv:
```powershell
.\opensimPy\Scripts\python.exe -m pip install <package>
```

Run the pipeline via `cli.py` (see `code/README.md` for the full command
reference):
```powershell
cd code
..\opensimPy\Scripts\python.exe cli.py preflight --subject AB08
..\opensimPy\Scripts\python.exe cli.py marker --subject AB08 --condition all
..\opensimPy\Scripts\python.exe cli.py calibrate --subject AB08
..\opensimPy\Scripts\python.exe cli.py framework --subject AB08 --condition all
..\opensimPy\Scripts\python.exe cli.py analyze --subject AB08
..\opensimPy\Scripts\python.exe cli.py correlate
```

Run the test suite (22 tests, ~7s):
```powershell
cd code
..\opensimPy\Scripts\python.exe -m pytest tests
```

## Data layout (`data/`)

```
data/
  subject_metadata.csv     SubjectID, Mass_kg, Height_m, ExoLeg, ShoeSize (5 or 7)
  opencap_sessions.csv     SubjectID, Condition, Activities, SessionID, Folder
                           (Folder blank == OpenCap session not yet downloaded)
  trial_status.csv         Subject, Trial, Process, Analysis, Reason (QC exclusions)
  Isokinetic_Data.xlsx     gold-standard isokinetic test results (see below)
  <SUBJECT>/
    RAW/MOCAP/*.c3d                    raw marker-based + force-plate capture
    RAW/INSOLE/*.txt                   raw insole pressure export
    RAW/OPENCAP/OpenCapData_<uuid>/    raw markerless mocap session
    PROCESSED/MOCAP/marker_data/*.trc  converted ground-truth markers
    PROCESSED/MOCAP/force_data/*.mot   converted ground-truth force-plate GRF
    PROCESSED/OPENCAP/OpenCapData_<uuid>/  mirrors RAW/OPENCAP, post-repair
```

Filename convention: `<SubjectID>_<Condition>_<ActivityToken>_<Side>_<TrialNumber>`,
e.g. `AB08_NOEXO_SLJ_R_3.c3d`. `Condition` is `NOEXO`/`EXO` (device worn or not --
see "Domain facts" below). SLJ's activity token is `SLJ`.

**Known data-quality issues the code is written to tolerate, not crash on**:
missing insole files for some trials, subjects in the metadata CSVs with no
`data/<ID>` folder yet, ambiguous/legacy filename suffixes, OpenCap sessions
whose `Folder` isn't populated yet. See "Never raise" under Architecture.

## Domain facts (non-obvious, load-bearing)

- Subjects are **able-bodied (AB)** recruits, not injured patients. `EXO`/`NOEXO`
  means the assistive device was worn or not -- it is **not** an injury/involved-limb
  axis. `EXO` trials exist **only for the right leg** by protocol.
- **LSI definitions** (`Config.lsi_definitions`) are therefore `NOEXO_L_vs_R`
  (natural left/right symmetry) and `NOEXO_L_vs_EXO_R` (device effect on the right
  leg relative to the natural left leg) -- verified numerically against
  `Isokinetic_Data.xlsx`: `PEAK_TORQUE_NOEXO_LSI` = `100 * LEFT_NOEXO/RIGHT_NOEXO`,
  `PEAK_TORQUE_WITHEXO_LSI` = `100 * LEFT_NOEXO/RIGHT_WITHEXO`, exactly.
- `Isokinetic_Data.xlsx`'s `SUBJECT_IDENTIFIER` uses unpadded IDs (`AB5`, `AB3`)
  while everywhere else uses zero-padded IDs (`AB05`, `AB03`). Always go through
  `subject.normalize_subject_id()` -- never compare raw ID strings.
- Only **AB03/AB05/AB07/AB08** currently have isokinetic data; **NTS01** has full
  movement data but no isokinetic row (useful for pipeline dev, excluded from the
  final correlation). All 5 subjects now have fully processed ground-truth +
  OpenCap + framework data (see Project status).
- **EXO model scaling reuses NOEXO's body-segment scale factors** (`manualScale`
  in the ScaleTool XML, factors carried over from the NOEXO anatomical run) and
  only re-runs `MarkerPlacer` against the EXO condition's own static trial --
  it does not recompute anatomical scaling. Verified empirically: NOEXO- and
  EXO-scaled models have byte-identical body segment masses; only marker
  placements differ.
- **Insole force-gain calibration must be fit per side** (`gain_L`, `gain_R`
  independently, each from that side's own trials) -- the old reference code's
  bug was fitting one shared gain from bilateral stance force and applying it to
  both feet. `subject_rules/*.json`'s `force_scale`/`force_scale_mode` are
  per-side dicts (`{"L": ..., "R": ...}`) for exactly this reason; the two
  existing subject files carry a placeholder (the old shared value duplicated
  into both sides, with a `force_scale_note` explaining it needs re-fitting).
- GRF-from-markerless accuracy (COM double-differentiation) has a **published,
  known limitation**: peak landing force is commonly biased >40% low, while
  propulsive-phase/impulse metrics stay reliable (sampling-rate mismatch +
  low-pass filtering killing the impact transient). See
  `resources/literature_notes.md`. The `insole_vertical_weight`/
  `com_vertical_weight` blend in the (not-yet-written) framework processor is a
  deliberate mitigation for this, not an arbitrary constant -- treat its weight
  as an investigation target, not a fixed default.

## Architecture

**`config.py`** -- one frozen `Config` dataclass, one instance (`DEFAULT`). Every
path is a `@property` derived from `data_root`/`project_root` (no legacy
uppercase module-level aliases, no hardcoded personal path). `data_root`
resolves via `RTS_DATA_ROOT` env var, else `<repo_root>/data`. `results_root`
is `<repo_root>/results` (i.e. `project_root.parent / "results"`, since
`project_root` is `code/`).

**`subject.py`** -- `Subject` (metadata + merged `subject_rules/<ID>.json`
overrides) and `Trial` (a plain dataclass of file paths for one trial).
`normalize_subject_id()` is the single place subject IDs get canonicalized.
Subject/Trial construction **never raises** for merely-absent data (a subject
with no folder, no metadata row, a missing file) -- that's `Preflight`'s job to
report, not an exception to throw mid-pipeline.

**`activities.py`** -- `Activity` is an ABC with two abstract methods,
`marker_plates(trial)` (which force plate(s) belong to which foot -- this is
genuinely activity-dependent) and `synchronize(trial, trc, ik_all, insole_all,
rules)` (activity-specific event detection, e.g. SLJ's flight-peak -> landing
-> ±0.75s window). Only `SingleLegJump` exists today. **Adding a new activity
means writing a new `Activity` subclass** -- `ForcePlateProcessor`,
`FrameworkProcessor` (not yet written), and `InsoleCalibration` (not yet
written) are meant to never change when that happens; they only ever call
`trial`'s activity via `activity.marker_plates(...)`/`.synchronize(...)`. An
`Activity` instance holds no per-subject state (rules are passed in
explicitly), so one instance is reusable across every subject.

**`discovery.py`** (`TrialDiscovery`) + **`preflight.py`** (`Preflight`) --
`TrialDiscovery.candidate_names()` globs raw C3D files unfiltered;
`.classify()` returns one of a fixed set of status strings (`OK`,
`NO_DATA_FOLDER`, `MISSING_MATCHING_INSOLE_FILE`, `UNRECOGNIZED_SUFFIX`,
`EXCLUDED_TRIAL_STATUS(<reason>)`, etc. -- see the module-level constants) for
one candidate name, **never raising**; `.trials()` returns only the `OK`,
non-excluded ones as real `Trial` objects for processing. `Preflight.run()`
walks every known subject (union of `subject_metadata.csv` rows and `data/`
folders) and writes one status row per subject/trial to
`results/_preflight/<activity>_trial_report.csv`, so a bad file surfaces as a
CSV line, not a stack trace three stages into a run.

**Ground-truth (marker-based) pipeline**, orchestrated by
**`marker_processor.py`**'s `MarkerProcessor.run(subject, activity, condition)`,
which loops conditions (`NOEXO`, `EXO`) **outside** the trial loop -- model
scaling happens once per condition, not once per trial:
1. `c3d_convert.py` (`C3dConverter`) -- raw C3D -> TRC + raw 6-plate force MOT
   via `opensim.C3DFileAdapter` (moving COP must be selected before reading).
2. `force_plates.py` (`ForcePlateProcessor`) -- right/left resultant GRF from
   the fixed or (for activities with no fixed plate) distance-inferred plates.
3. `scale_ik_id.py` -- `ModelScaler.scale()` (measurement-based for NOEXO,
   `manualScale` reusing NOEXO's factors for EXO -- see Domain facts),
   `InverseKinematicsRunner.run()`, `InverseDynamicsRunner.run()` (right force
   is always `ground_force_1` -> `calcn_r`, left is always `ground_force_2` ->
   `calcn_l`). All three build OpenSim setup XML via string substitution
   (there's no friendlier builder in OpenSim's API) -- kept isolated to this
   one module.

**`io_tables.py`** -- all MOT/STO/TRC read/write goes through OpenSim's own
`TimeSeriesTable`/file adapters, never a hand-written parser; the raw insole
`.txt` format is the one thing read with plain pandas (column names located by
substring match, since header wording has varied across export sessions).

**`insole_geometry.py`** (`InsoleGeometry`) -- foot marker lookup and the
anatomical-frame construction (`frames()`, `foot_height()`, `marker()`) are
classmethods usable without a shoe size (so `activities.py` can call them
directly); only the native-COP <-> lab-frame transform needs an instance
(`InsoleGeometry(shoe_size)` or `.for_subject(subject)`), since only that needs
the measured insole dimensions.

**`opensim_env.py`** -- the one place `import opensim` happens; sets the
OpenSim logger level from config. Also owns `center_of_mass()` (per-frame
whole-body COM from an IK motion, shared by the framework pipeline and
`trc_repair.py`'s validator).

**Framework (markerless) pipeline**: **`calibration.py`** (`InsoleCalibration`)
fits `gain_L`/`gain_R` independently (see Domain facts) plus a per-side rigid
COP offset, once per subject, written to
`results/calibration/<ID>_insole_marker_foot_calibration.json`.
**`framework_processor.py`** (`FrameworkProcessor`) reads OpenCap TRC/IK +
insole, calls `activity.synchronize()`, computes COM via
`opensim_env.center_of_mass()`, double-differentiates for total GRF, splits
it into right/left using the insole force fraction and the per-side
calibration, and runs framework ID via `scale_ik_id.InverseDynamicsRunner
.run_with_paths()` (the same generic ID-XML-building method the marker
pipeline's `InverseDynamicsRunner.run()` also calls, refactored out so both
pipelines share one bilateral-force-convention implementation).

**Outcomes, LSI, correlation**: **`outcomes.py`** (`OutcomeExtractor`)
compares framework vs. marker GRF/COP/moments/work per trial ->
`trial_summary.csv`. **`lsi.py`** (`LsiCalculator`) applies
`100 * mean(test)/mean(reference)` per `config.lsi_definitions` ->
`lsi_summary.csv`. **`isokinetic.py`** (`IsokineticTable`) loads/normalizes
`Isokinetic_Data.xlsx`. **`correlation.py`** (`CorrelationAnalysis`) joins
SLJ LSI to isokinetic LSI via each `lsi_definitions` entry's *reference*
condition mapped through `config.isokinetic_condition_map` (not a second
hardcoded mapping), and via a fixed SLJ-metric -> isokinetic-metric analogy
(`PeakKneeMoment` -> `peak_torque`, `KneeAbsoluteWork` -> `total_work`).
Always reports `n`; never assumes a subject count. Besides the aggregate,
cross-subject `run()` (writing to `results/_correlation/`), it also has
`subject_report(subject, activity)` -- reuses the same `build_join()` logic
scoped to one subject, writing to that subject's own
`results/<SUBJECT>/analysis/<activity>/isokinetic_association/` folder (a
bar chart of framework/marker/isokinetic LSI side by side, plus the raw
join CSV). `project.py`'s `run_analysis()` calls this automatically after
outcomes/LSI on every `analyze`/`framework` CLI run, so it never needs a
separate step. Writes a header-only CSV (no plot) for a subject with no
isokinetic row (e.g. NTS01) rather than nothing, so a human sees *why*
there's no association, not a missing file.

**`project.py`** (`RtsAssessmentProject`) is the facade wiring one `Subject`
to every processor above; **`cli.py`** is the argparse entry point
(`preflight | marker | calibrate | framework | analyze | correlate`).

**`opencap_admin/download.py`** (`OpenCapDownloader`) pulls one subject's
OpenCap sessions via its REST API (credentials file, default
`~/Documents/opencap-processing/.env`), then mirrors `RAW/OPENCAP` ->
`PROCESSED/OPENCAP` (the repair-ready copy) for any session folder not
already mirrored.

**`trc_repair.py`** -- marker-trajectory QC, ported from the in-progress
prototype (see Project status below for the exact source path). Three
explicit, human-gated CLI steps (`cli.py repair-filter|repair-validate
|repair-promote --subject <ID> --trial <name>`), none of which affects
production output except the last:
1. `AutomaticTrcFilter.process_trial()` -- batch-filters a whole trial's
   markers (10 Hz zero-phase Butterworth) and flags unusually large
   per-marker change for review, writing a candidate repaired TRC.
2. `TrcRepairValidator.validate()` -- reruns OpenCap's own IK setup
   (official marker weights, patella bodies/joints/constraints removed
   exactly as OpenCap's IK does, via the shared **`opencap_ik.py`**
   (`OpenCapIkRunner`) -- extracted standalone specifically so both
   `trc_repair.py` and `framework_processor.py` can import it without a
   circular dependency) and the framework pipeline on both the original
   and the candidate repaired TRC, then compares hip/knee/ankle angle and
   moment against the marker/ground-truth pipeline -- so a repair's effect
   is never conflated with a different pipeline run. Prints a
   baseline-vs-repaired RMS table and a comparison plot; never promotes
   anything itself.
3. `RepairPromoter.promote()` -- the **only** step that makes a repair take
   effect: adds the trial to `subject_rules/<ID>.json`'s `repaired_trials`
   list and copies the filtered TRC into
   `PROCESSED/OPENCAP_REPAIRED/<name>.trc`. Only run this after reviewing
   step 2's output and judging it a real improvement.

Once promoted, `Trial.effective_opencap_ik()` (`subject.py`) and
`FrameworkProcessor.process()` automatically **rerun OpenCap's IK from the
repaired TRC** (via the same `OpenCapIkRunner`) before framework processing
uses it -- a repaired TRC is never combined with the original, stale IK
output.

## Project status (what exists vs. what's still to be built)

**The entire SLJ pipeline is implemented and runs end-to-end against real
data**, verified on AB08 (12 trials, both conditions): ground-truth pipeline
(scale/IK/ID), per-side insole calibration, framework pipeline (bilateral
GRF/COP + ID), outcome extraction, LSI computation, and the isokinetic
correlation join. `python cli.py preflight|marker|calibrate|framework|analyze|correlate`
all work -- see `code/README.md` for exact commands. 22 tests pass
(`python -m pytest tests`), including real-data regression tests in
`test_discovery.py`/`test_isokinetic.py` and synthetic ones for
`activities.py`/`calibration.py`/`insole_geometry.py`/`lsi.py`.

All modules named in "Architecture" above now exist: `config.py`,
`subject.py`, `discovery.py`, `preflight.py`, `activities.py`, `io_tables.py`,
`opensim_env.py`, `insole_geometry.py`, `c3d_convert.py`, `force_plates.py`,
`scale_ik_id.py`, `marker_processor.py`, **`calibration.py`**
(`InsoleCalibration`, per-side gain + rigid COP offset -- verified against
real AB08 insole data to recover genuinely distinct `gain_L`/`gain_R`),
**`framework_processor.py`** (`FrameworkProcessor` -- OpenCap+insole ->
bilateral GRF/COP -> ID; peak vGRF came out at ~2.8x body weight for
AB08's first trial, within the physiologically plausible 1.5-4x range),
**`outcomes.py`** (`OutcomeExtractor`), **`lsi.py`** (`LsiCalculator`),
**`isokinetic.py`** (`IsokineticTable`), **`correlation.py`**
(`CorrelationAnalysis`), **`project.py`** (`RtsAssessmentProject` facade),
**`cli.py`**, **`opencap_admin/download.py`** (`OpenCapDownloader` -- a real
OpenCap credentials file already exists on this machine at the default path
`~/Documents/opencap-processing/.env`, so this has been exercised against
the real API, not just written), and **`trc_repair.py`** (`AutomaticTrcFilter`
+ `TrcRepairValidator`, ported from the in-progress prototype at
`G:\Shared drives\SuhrudLib\PILOT_OOP_BASE\temp\AB08_SLJ_moment_investigation\trc_repair_gui\`
-- the automatic batch filter and the matched original-vs-repaired
validation math, **not** the interactive manual-review GUI, which remains
unported).

**All 5 subjects (AB03/AB05/AB07/AB08/NTS01) now have complete pipelines
run** -- ground truth + framework + calibration + LSI for every one, and
`cli.py correlate` produces the real framework-vs-isokinetic join across all
4 subjects with isokinetic data (AB03/05/07/08), n=4, reported honestly (weak
correlations at this n -- expected, not a bug; never hardcode n anywhere).

**Two real data-integrity issues found and fixed via `opencap_aliases`** (not
papered over -- both were verified against `sessionMetadata.yaml`'s own
`subjectID`/`mass_kg`/`height_m` fields before concluding they were safe to
alias, not a genuine cross-subject data mixup):
- **AB03's NOEXO session** trial files are misnamed `ABO3_...` (letter O)
  instead of `AB03_...`. `subject_rules/AB03.json` maps every canonical NOEXO
  SLJ trial name to its actual filename. `AB03_NOEXO_SLJ_L_2` has no
  OpenCap trial at all (confirmed genuinely missing, not a naming issue).
- **AB07's NOEXO session** trial files are labeled `AB08_...` instead of
  `AB07_...` -- confirmed genuinely AB07's own recording (its
  `sessionMetadata.yaml` says `subjectID: AB07, mass_kg: 64.4, height_m:
  1.753`, matching `subject_metadata.csv` exactly and clearly not AB08's
  62.6kg/1.6256m) -- an operator typed the wrong subject ID as the trial-name
  prefix during capture. `subject_rules/AB07.json` has the alias map.

**`opencap_admin/download.py` gained retry logic** after two subjects'
downloads hit the same transient `RemoteDisconnected` error on their second
session's info request -- `_with_retries()` (3 attempts, backoff) now wraps
both the session-info GET and each media download, and `download_subject()`
no longer lets one session's failure lose progress on a subject's other
sessions.

**`framework_processor.py`/`marker_processor.py`'s `.run()` methods now skip
a trial with a known-missing file (logged, not silent) instead of aborting
the whole batch** -- found for real when AB03's genuinely-missing
`SLJ_L_2` OpenCap trial crashed a 16-trial framework run.

**GRF blend-weight investigation done** (see
`code/expt_temp/grf_blend_weight_sweep/FINDINGS.md` and
`resources/literature_notes.md`): for AB08, pure insole
(`insole_vertical_weight=1.0`) gave the lowest peak-vertical-GRF bias (-3.1%)
of any weight tried -- preliminary, n=1 subject, peak GRF only.

**2026-09-11: ported the upstream `PILOT_OOP_BASE_PY`'s "improved GRF
estimation," applied only to AB08 so far** (the live upstream repo at
`G:\Shared drives\SuhrudLib\PILOT_OOP_BASE_PY` had moved ahead of the
`resources/_old_reference_code` snapshot -- always check the live repo, not
just the snapshot, before assuming a module is a faithful copy):

- **Shared "Lai" model** (`ModelScaler.prepare_lai_model()` in
  `scale_ik_id.py`): the marker-based AND framework pipelines now use ONE
  model per subject/condition, derived from the subject's own OpenCap-scaled
  model (consistent body-segment scaling), with body masses/inertias
  corrected to exactly match the subject's *measured* mass (OpenCap's own
  video-based mass estimate doesn't always match it), and markers replaced
  from the study's reference model wherever they exist in the subject's
  static trial. This **replaces** `ModelScaler.scale()` (Biomech57 anatomical
  scaling) as what `MarkerProcessor.run()` actually uses -- `scale()` is kept
  as a still-valid utility, just no longer on the default path. **New
  dependency**: marker-based processing now requires an accepted trial's
  OpenCap model to exist, since the shared model derives from it.
- **Rotation-force (H-dot) correction for horizontal GRF** (`opensim_env
  .center_of_mass_and_momentum_rate()` + `FrameworkProcessor._bilateral_forces
  ()`): horizontal COM double-differentiation alone is known to be especially
  noisy from video-based kinematics. When `activity.name` is listed in a
  subject's `rotation_force_activities` (and `activity.unilateral_framework`),
  Fx/Fz are blended with a moment-balance estimate -- the horizontal force
  implied by the body's own filtered angular-momentum rate about the observed
  COP -- weighted by `rotation_force_weight_x`/`_z`. `subject_rules/AB08.json`
  now has `rotation_force_activities: ["single_leg_jump"]`, both weights 0.5,
  matching the tuned values already in the upstream repo. `default.json` and
  every other subject default to no correction (empty list, weight 0).
- **Enhanced per-trial comparison plot** (`OutcomeExtractor._plot_trial`,
  `<trial>_sync_grf_cop_angles_id.png`): now a 4x3 grid -- Fx/Fy/Fz GRF, AP/ML
  COP, support moment (Winter's whole-limb extensor-moment sum, a new
  `outcomes.py`/`lsi.py` metric ported from upstream too), hip/knee/ankle
  angle, hip/knee/ankle moment -- every panel except the two angle rows
  titled with the framework-vs-marker RMS. This is the main visual
  per-trial QC output; regenerated for all 12 AB08 trials in
  `results/AB08/analysis/single_leg_jump/trial_comparisons/`.

**2026-09-16: all 5 subjects (AB03/AB05/AB07/AB08/NTS01) fully reprocessed
with the shared-Lai-model pipeline** (marker -> framework -> outcomes -> LSI
-> isokinetic association, `overwrite=True` to force recomputation) --
AB03/AB05/AB07/NTS01 previously reflected the old Biomech57-scaled,
translation-only-horizontal-force pipeline; all are now on the same
pipeline as AB08 and directly comparable. Note: only **AB08** has
`rotation_force_activities` configured in `subject_rules/AB08.json` (the
H-dot horizontal-force blend) -- AB03/AB05/AB07/NTS01 all use
`default.json`'s empty list, so their Fx/Fz are translation-only. This is a
known, deliberate asymmetry (not yet investigated per-subject) to keep in
mind when comparing horizontal-GRF-derived metrics across subjects.

Two real issues surfaced and fixed **during** this rerun (both found on
NTS01, neither a regression from the shared-model port itself):
- **`FrameworkProcessor.run()` only caught `FileNotFoundError`, not a
  genuine per-trial `ValueError`** raised by `activity.synchronize()` when
  the insole and IK/TRC time windows never overlap (`NTS01_NOEXO_SLJ_L_1`,
  confirmed pre-existing -- this trial was never previously in any earlier
  framework output either) -- now caught and skipped the same way, logged
  not silent, so one bad trial can't abort a whole subject's batch.
- **NTS01 had never had `InsoleCalibration.fit()` run** (no
  `results/calibration/NTS01_insole_marker_foot_calibration.json` existed
  before this rerun) -- fit now exists; NTS01's framework pipeline
  completed 11/12 trials (the one skip is the synchronization failure
  above).

**Padding fix for the `np.gradient` double-differentiation boundary
artifact** (`config.com_gradient_pad_s`, `FrameworkProcessor._padded_time()`):
`np.gradient` uses lower-order one-sided differences at the first/last
sample of an array, which compounds severely under double differentiation
(COM position -> velocity -> acceleration), producing a sharp spurious
spike at the very start/end of the reported window. Fixed by differentiating
over an artificially extended time window (padded by `com_gradient_pad_s`,
clamped to the IK's own available time range) and cropping back to the
reported window afterward, so the one-sided-difference error lands in the
discarded padding, not the reported signal. Distinct from a broader,
still-present dip seen mid-push-off in some AB08 rotation-force-blended
trials -- that is a genuine (not yet fully vetted) H-dot-driven signal
change, not this boundary artifact; don't conflate the two.

**Still to do**: review the new AB08 plots for issues (a large, consistent
hip-angle offset between framework and marker is visible in at least one
trial -- worth checking whether it's a genuine model/pose-convention
difference or a real bug before trusting hip-related metrics), decide
whether `rotation_force_activities` should be extended to AB03/05/07/NTS01
now that they're on the shared-model pipeline, check the GRF blend-weight
finding against a second subject/metric, the interactive TRC-repair GUI,
and eventually a second `Activity` subclass.

`code/expt_temp/` is a sanctioned scratch area for trying out new techniques in
isolation (see its `README.md`) -- nothing there is imported by production code.
