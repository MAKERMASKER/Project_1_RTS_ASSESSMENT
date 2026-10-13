# OpenSimAD objective-weight search

Tests whether OpenSimAD's own trajectory-optimization objective weights
(specifically `accelerationTrackingTerm` and `activationTerm`) are
responsible for the systematic GRF underestimation / non-physical early
"dip-then-rise" wobble found in
`code/expt_temp/opensimad_gain_check/synced_comparison_plots/`.

Every objective term is already overridable via `settings['weights']`
(confirmed by reading `mainOpenSimAD.py` directly) -- no edits to the
vendored OpenSimAD repo needed. `run_weighted_trial.py` applies a JSON
weight override to the `settings` dict `processInputsOpenSimAD()` returns,
right before calling `run_tracking()`.

Pilot trial: `AB03_NOEXO_SLJ_R_1` (side R, push-off window `[3.24, 3.84]`
-- the most-examined trial this session, with an existing converged
baseline result and synced comparison plot already available).

## Configs tested

'jumping' preset baseline (`settingsOpenSimAD.py:587-597`):
`positionTrackingTerm=100, velocityTrackingTerm=10,
accelerationTrackingTerm=50, activationTerm=1`.

| Config | accelerationTrackingTerm | activationTerm | Rationale |
|---|---|---|---|
| config0_baseline | 50 | 1 | current default (reused, not rerun) |
| config1_accel500 | 500 (10x) | 1 | stiffer acceleration tracking alone |
| config2_activation01 | 50 | 0.1 (10x lower) | relaxed effort penalty alone |
| config3_combined | 500 | 0.1 | combined effect |

## Files

- `run_weighted_trial.py` -- parametrized OpenSimAD driver with weight
  override support (conda env subprocess, mirrors
  `../opensimad_gain_check/run_one_trial.py`).
- `run_configs.sh` -- runs configs 1-3 sequentially for the pilot trial.
- `evaluate_configs.py` -- outer objective: whole-trace RMSE of each
  config's OpenSimAD GRF vs. real marker ground truth (same methodology
  as `../opensimad_gain_check/analyze_pushoff_rms.py`) -> `results/
  config_ranking.csv`.
- `plot_configs.py` -- overlays ground truth + framework/production +
  all 4 configs on one plot -> `results/
  AB03_NOEXO_SLJ_R_1_all_configs_comparison.png`.

## Results

| Config | Status | RMSE vs marker (%) | Gain gap |
|---|---|---|---|
| config3_combined (accel=500, activation=0.1) | non-convergent | 42.4% | -15.4% |
| config1_accel500 | non-convergent | 42.5% | -15.6% |
| config0_baseline (accel=50, activation=1) | converged | 42.8% | -17.8% |
| config2_activation01 | converged | 50.2% | -20.6% |

**Negative result: neither knob fixes the wobble.** The comparison plot
(`results/AB03_NOEXO_SLJ_R_1_all_configs_comparison.png`) shows all 4
configs reproducing the SAME early dip-then-rise artifact (drop to
~930-1000N around t=0.08s, rise back to ~1500-1530N by t=0.15-0.19s) at
nearly identical magnitude and timing -- `accelerationTrackingTerm` 50 vs
500 barely changes the curve shape at all (visually near-overlapping),
despite the 10x change. `activationTerm` 0.1 vs 1 is the only
visually-distinct curve, but it's WORSE, not better: an earlier, sharper,
non-physical drop-off in the late portion of the window, matching its
worse RMSE (50.2%, the worst of the 4). The RMSE differences between
configs are marginal (all within ~8 points of each other) and 2 of the 3
non-default configs (both including `accelerationTrackingTerm=500`)
failed to converge cleanly.

**Conclusion**: the wobble is not explained by the position/acceleration-
tracking-vs-muscle-effort balance in the range tested. Its root cause is
likely elsewhere (input-kinematics filtering/spline-fitting of the
tracked accelerations, the foot-ground contact model's own dynamics
during the transition into/out of contact, or something structural in
the trajectory optimization's handling of that transition) -- not
further explored in this pass.
