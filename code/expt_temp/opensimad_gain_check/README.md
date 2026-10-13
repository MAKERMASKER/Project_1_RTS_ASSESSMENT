# OpenSimAD kinematics-only GRF: sensor-gain check

Investigates whether OpenSimAD (a full musculoskeletal trajectory-
optimization tracking engine, GRF as a byproduct of physics rather than a
tracked external signal) can supply a marker-ground-truth-free reference
for fitting each subject's insole vertical-force gain -- the user's own
proposed alternative to quasi-static stance calibration (see this
session's insole-gain investigation: stance-based gain doesn't correlate
with body weight or shoe size, at any low-pass cutoff tried).

## What "how OpenSimAD is being run" actually means here (read from
`G:\Shared drives\SuhrudLib2\Projects\Project_1_side_GRF\code`, nothing
else from that project)

- It is **not** run inside this project's `opensimPy` venv. It requires a
  separate conda environment (`opencap-processing-grf`, already installed
  at `C:\Users\sjogleka\AppData\Local\anaconda3\envs\opencap-processing-grf`)
  and a separate vendored repo clone at
  `C:\Users\sjogleka\Documents\opencap-processing-grf-jumptest\` (a fork of
  stanfordnmbl/opencap-processing with jump/GRF-tracking support, per
  github.com/utahmobl/opencap-processing-grf).
- Entry point: that repo's own `run_jump_ad.py` (three steps --
  `processInputsOpenSimAD` builds the muscle-wrapping/contact/external-
  function model **once per session** [not per trial -- reused across a
  session's other trials], `run_tracking` runs the actual IPOPT
  trajectory-optimization solve for one trial's time window, then
  `plotResultsOpenSimAD`), with `motion_type="jumping"`.
- Requires OpenCap API authentication at import time -- already satisfied
  by an existing `.env`/`API_TOKEN` in that repo.
- **Real, documented cost** (from Project_1_side_GRF's own single
  exploratory run, `AB03_EXO_SLJ_R_1`, ~1.5s window): **~2.5 hours of
  solver time, and it did NOT converge** (`Maximum_Iterations_Exceeded`
  at IPOPT's 5000-iteration cap), even after manually widening two
  hardcoded, gait-oriented ROM safety caps in the vendored repo's
  `mainOpenSimAD.py` (`ankle_angle_*` 50->60deg, `subtalar_angle_*`
  35->45deg) that otherwise clipped the polynomial-fitting bounds below
  this subject's actual landing kinematics. Peak vertical GRF was
  plausible despite non-convergence; Fx/Fz showed spurious oscillations
  at impact, a visible symptom of an incomplete solve. That project
  deliberately did not scale this beyond one trial given the cost.

## This folder's scope (per the user's request)

ONE NOEXO SLJ trial per subject/side, landing peak +/- 0.3s (shorter than
the ~1.5s window tried before, to see if a much shorter horizon is more
tractable) -- a timed pilot first, not a blind full sweep, given the
documented 2.5-hour/non-convergent precedent above.

Files:
- `prepare_session_data.py` -- copies one subject/condition's OpenCap
  session folder (MarkerData/OpenSimData/Model/sessionMetadata.yaml) from
  this project's `data/<SUBJECT>/RAW/OPENCAP/<session>/` into the external
  repo's `Data/<session>/`, if not already mirrored there. Pure file copy,
  not compute.
- `run_one_trial.py` -- parametrized version of the reference repo's own
  `run_jump_ad.py`, run via subprocess against the `opencap-processing-grf`
  conda env's own python.exe (cannot be imported directly -- different
  Python/CasADi/IPOPT install from this project's `opensimPy` venv).
- `plan_trials.py` -- resolves "one NOEXO trial per subject/side" +
  landing-peak +/-0.3s windows from this project's own framework
  trial_data.npz (peak_grf_time_s), without guessing.
- `results/<trial_name>/` -- OpenSimAD's own output (GRF.csv, plots, solver
  log), one subfolder per trial, plus `timing.txt` (wall-clock duration)
  and `status.txt` (converged / did not converge / crashed).
