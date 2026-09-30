# OpenSimAD as a marker-ground-truth-free insole gain reference -- investigation log

Standing reference for this line of investigation, so it doesn't live
only in chat. Goal: find a way to calibrate each subject's insole
vertical-force gain (`subject_rules/*.json`'s `force_scale`) *without*
needing marker/force-plate ground truth -- useful for eventual field
deployment (OpenCap + insole only). Paused here as of 2026-09-24; revisit
with the open threads at the bottom.

All work lives under `code/expt_temp/`:
- `opensimad_gain_check/` -- landing-window and push-off-window pilots,
  the full-batch push-off run, the corrected whole-trace RMS
  methodology, the uniform-correction test, and the full-pipeline
  `GrfRmse_N` validation.
- `OpensimAD_weights_optimization/` -- the objective-weight search.

## 1. What OpenSimAD is and how it's run

A separate, external tool (not part of this project's own `opensimPy`
venv): a musculoskeletal trajectory-optimization engine that tracks
OpenCap kinematics through a full dynamic simulation with a physical
foot-ground contact model, producing GRF as a byproduct of the physics
(not a tracked external signal). Runs in a separate conda env
(`opencap-processing-grf`) against a separate repo clone
(`C:\Users\sjogleka\Documents\opencap-processing-grf-jumptest\`, forked
from stanfordnmbl/opencap-processing). Invoked via
`conda run -n opencap-processing-grf python <driver>.py` (subprocess --
different Python/CasADi/IPOPT install). Every objective-function weight
is overridable via a `settings['weights']` dict returned by
`processInputsOpenSimAD(...)`, mutated before calling `run_tracking(...)`
-- no edits to the vendored repo needed for weight experiments.

## 2. Approaches tried, in order

### 2.1 Landing-window pilot -- failed outright
Window: landing peak +/- 0.3s (the natural first choice, matching the
event framework/marker pipelines already anchor on). Result: **did not
converge** even after ~2.5 hours of solver time (`Maximum_Iterations_
Exceeded`), matching a prior, independent finding from a sibling project
(`Project_1_side_GRF`) that hit the identical failure mode on the same
kind of window. Conclusion: the sharp, fast landing transient is too
demanding for this solver/window combination.

### 2.2 Push-off-window pilot -- converges, but see 2.3
Window: liftoff ("insole force returns to zero going into flight") +/-
0.3s -- a smoother, less impulsive event than landing. Converges
reliably in ~30-50 min/trial (vs. 2.5h+/non-convergent for landing).
Initial (later shown to be misleading) peak-only comparison suggested
~9% mean error against marker ground truth across 9 trials -- see 2.3
for why this was wrong.

### 2.3 Correcting the comparison metric -- peak-only was misleading
User caught that comparing only the SINGLE MAXIMUM sample of each trace
(OpenSimAD vs. marker) can agree or disagree almost by coincidence while
the rest of the curve disagrees badly. Rebuilt the check as whole-trace
RMSE (both signals resampled onto a common time grid across the full
push-off window) plus a least-squares (regression-through-origin) gain
fit instead of a peak ratio. Result, same 9-10 trials: **RMSE 38-115%
(mean ~63%)**, and the OpenSimAD-implied gain is **systematically biased
low, every single trial, no exceptions** (-5% to -68%, mean ~-20%, or
~-13.5% std 4.3% if the one ROM-cap-limited outlier trial is excluded).
The synced 3-way comparison plots
(`opensimad_gain_check/synced_comparison_plots/results/`) show WHY: a
real, non-physical early "dip-then-rise" wobble in OpenSimAD's estimate
that a peak-only check completely hides.

### 2.4 Uniform correction factor -- tested, rejected
Field deployment can't cherry-pick per subject (no ground truth to know
who needs correcting) -- tested whether a single fixed multiplier
(1.156, from the ~-13.5% mean gap above) applied to every subject
improves the REAL, full-pipeline metric this project trusts (`GrfRmse_N`
from `outcomes.py`, whole synced window, contact-masked -- not the ad
hoc window-only trace RMSE). Result (`uniform_correction/results/
uniform_correction_summary.csv`): only 2/8 trials improved (the two
AB03 trials, whose starting calibration was worst) -- the other 6 got
substantially worse, some dramatically (AB10/L peak GRF overshot from a
reasonable 2980N to 4003N against a true 2618N). **Rejected**: helps
only the subject with the worst starting point, hurts everyone else.

### 2.5 Cross-side gain application -- mixed
Tested whether one side's OpenSimAD-derived gain (whichever side
converged with lower error) works when applied to the OTHER side of the
same subject -- a `single per-subject gain` simplification. 3/4 subjects
improved (sometimes dramatically, AB03: -21.8% -> -6.5%), 1/4 (AB10) got
worse. Re-tested AB10 specifically with two BRAND NEW trials (not
trial-selection luck) -- same result both times (~-14 to -18% gap,
consistent) -- confirmed genuine, not an artifact of which trial was
picked.

### 2.6 Objective-weight search -- negative result (2026-09-24)
Hypothesis: the wobble/underestimation comes from OpenSimAD's own
objective balance between kinematic tracking stiffness
(`accelerationTrackingTerm`) and muscle-effort penalty
(`activationTerm`). Tested 3 configs (10x accel alone, 10x-lower
activation alone, both combined) against the `'jumping'` preset's
defaults (accel=50, activation=1) on the single best-understood pilot
trial (`AB03_NOEXO_SLJ_R_1`). **All 4 configs show the identical wobble
shape and timing** -- `accelerationTrackingTerm` 50 vs. 500 barely moves
the curve at all despite the 10x change; lowering `activationTerm` is
the only visually distinct result, and it's WORSE (a new, sharper,
non-physical late-window drop-off), matching its worse RMSE (50.2%, the
worst of the 4). RMSE spread across all 4 configs is small (~8 points)
and 2 of 3 non-default configs failed to converge cleanly. **Conclusion:
this specific hypothesis is not supported** -- the wobble's root cause
is elsewhere (candidates: input-kinematics filtering/spline-fitting of
tracked accelerations, the foot-ground contact model's own dynamics
during the stance/flight transition, or something structural in how the
optimization handles that transition).

## 3. Open threads for when this resumes

- The wobble's real root cause is still unknown -- filtering cutoff
  frequency (`settingsOpenSimAD.py`'s `cutoff_freq_Qs/Qds/Qdds`, all 10
  Hz for the `'jumping'` preset) and the contact model are the next
  candidates, not yet tested.
- A completely different, much cheaper technique was proposed and not
  yet tried: flight-time/impulse-momentum calibration (takeoff velocity
  from kinematics alone via `v = 0.5*g*t_flight`, cross-referenced
  against the raw insole's own integrated impulse) -- no expensive
  trajectory optimization needed at all.
- A much more extreme weight change (100x rather than 10x) was suggested
  but not tried before this was paused.
- Related, separate finding surfaced during this work (not an OpenSimAD
  issue, documented in `CLAUDE.md`'s Domain facts instead): the
  marker-mocap lab's global frame and OpenCap's own global frame are
  rotated ~90deg apart -- fixed in `outcomes.py`'s Fx/Fz comparison,
  unrelated to the gain-calibration question above.
