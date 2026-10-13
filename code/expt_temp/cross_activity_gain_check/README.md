# Cross-activity insole gain check -- does one gain serve every activity?

## Question

The production gain (`vertical_force_scale`) is fit **once per subject/side
from SLJ quiet single-leg stance only** -- `calibration.py`'s `__init__`
hardcodes `activity_for("single_leg_jump")` -- and then applied unchanged to
every activity. Is that defensible? If SLJ's stance gain is 0.78 and SLJ's
ideal is 0.82, does 0.82 still hold for lateral hop and skater hop?

## Method

For every trial, fit the gain that makes raw insole vertical force best match
the force plate -- the best any single scalar gain could possibly do on that
trial. Regression through the origin (the estimator already established in
this project; a peak ratio is reported for reference but never used as the
fit, since single-point matching was previously shown to be misleading):

```
ideal = sum(insole * plate) / sum(insole ** 2)
```

Because the cross term vanishes for a through-origin fit, the cost of using
any *other* gain `g` is exact:

```
RMSE(g)^2 = RMSE(ideal)^2 + (g - ideal)^2 * mean(raw^2)
```

so the added error is `|g - ideal| * RmsRawInsole_N`. That is how the transfer
cost below is computed -- analytically, not by re-fitting.

**These gains are ground-truth-derived and NOT field-deployable.** They need a
force plate, which is precisely what the framework pipeline exists to avoid.
They are a BENCHMARK/CEILING ("how good could the gain possibly be") and a
test of transferability. Any run using them is a benchmark configuration, not
evidence of the markerless pipeline's independent accuracy.

Axis note: vertical force (Fy) only. The ~90deg marker-lab vs. OpenCap global
frame rotation documented in `CLAUDE.md` affects Fx/Fz exclusively and is
irrelevant here; Fy is common to both frames. (Audited: `outcomes.py` is the
only production code comparing raw marker Fx/Fz, and it applies the correction
correctly -- `opencap_X = +marker_Z`, `opencap_Z = -marker_X`.)

Triple hop is deliberately excluded: its state detector is not yet finalised,
so its insole<->OpenCap sync cannot be trusted for a gain fit.

## Results (2026-09-28; 135 (trial, side) fits, 6 subjects, 3 activities)

Trials the activity's own sync flags `quality_omit` are excluded, matching the
analysis stage's gate -- see "The quality_omit gate" below.

### The gain transfers. The activities do not.

| Activity | own-activity ideal gain | SLJ-derived gain | current production gain | n |
|---|---|---|---|---|
| SLJ | **116 N (6.1% of peak)** | 119 N (6.3%) | 149 N (8.0%) | 84 |
| LH  | **469 N (25.7%)** | 481 N (26.3%) | 515 N (27.4%) | 31 |
| SKH | **243 N (12.5%)** | 266 N (15.9%) | 298 N (16.4%) | 20 |

- **Transferring the SLJ gain to LH/SKH costs a median of 16 N** added RMSE --
  negligible against the 240-470 N that remains even with a perfect
  per-activity gain.
- There **is** a real systematic shift: LH/SKH ideal gains sit **~10% below**
  SLJ's, in **37 of 51** (trial, side) fits. Real and consistent, but small in
  its effect on error.
- Moving from the current stance gain to ideal gains helps modestly
  everywhere (~30-55 N), and most in relative terms on SLJ (8.0% -> 6.1%).

### The headline

**Lateral hop carries ~26% error with a mathematically perfect gain.** No gain
fixes that, because it is not gain error -- it is shape/sync/contact error.
Chasing gain precision for LH is optimising the wrong term. SLJ at 6.1% is
genuinely good; SKH at 14.3% is intermediate.

### Median ideal gain per subject/side/activity

| Subject/Side | SLJ | LH | SKH | current | LH-SLJ | SKH-SLJ |
|---|---|---|---|---|---|---|
| AB03 L | 0.892 | 0.905 | -- | 0.681 | +0.012 | -- |
| AB03 R | 0.933 | 0.810 | 0.798 | 0.690 | -0.123 | -0.135 |
| AB05 L | 0.737 | 0.625 | -- | 0.936 | -0.112 | -- |
| AB05 R | 0.794 | 0.764 | 0.662 | 0.953 | -0.030 | -0.132 |
| AB07 L | 0.966 | 0.926 | -- | 0.978 | -0.040 | -- |
| AB07 R | 1.042 | 0.848 | 0.987 | 1.028 | -0.194 | -0.055 |
| AB08 L | 0.712 | 0.601 | -- | 0.616 | -0.111 | -- |
| AB08 R | 0.727 | 0.677 | 0.545 | 0.651 | -0.050 | -0.181 |
| AB10 L | 1.107 | 1.048 | -- | 1.111 | -0.058 | -- |
| AB10 R | 1.137 | 1.019 | 1.070 | 1.099 | -0.117 | -0.067 |
| NTS01 L | 0.820 | 0.620 | -- | 0.857 | -0.200 | -- |
| NTS01 R | 0.850 | 0.372 | 0.461 | 0.794 | **-0.478** | **-0.390** |

Within-activity trial-to-trial spread (IQR) is tight for SLJ (0.010-0.054)
and much looser for LH (0.054-0.159) and SKH (0.086-0.502) -- another sign
that the hop activities are less well-conditioned, not just differently
scaled.

(The per-subject table above is from the first pass, which still included
NTS01's `quality_omit` LH/SKH trials -- those NTS01 LH/SKH entries are the
discredited ones; every other row is unchanged by the fix.)

### The quality_omit gate -- a real bug this check surfaced

The first pass fit gains on ANY trial that synchronised, and produced a
confident-looking NTS01 LH/R "ideal" gain of **0.372** against that subject's
SLJ value of 0.850 -- a factor of 2.3.

That number was not a gain. **Every one of NTS01's LH and SKH trials is
flagged `quality_omit=True` by the activity's own sync detector**, and
`outcomes.py` refuses to analyse them (`if bool(data.get("quality_omit"))
: return None`). The gain fitter did not apply that same gate, so it happily
fit a scalar to mis-aligned data. Three independent signals agreed once the
gate was checked: the outlier gain, the sync flag, and the analysis refusal.

`fit_ideal_gains.py` now honours `sync["quality_omit"]`. Effect of the fix:
140 -> 135 fits; the 5 dropped are exactly NTS01's 3 LH + 2 SKH. **No other
subject was affected** -- their `quality_omit` trials were already filtered
out upstream by `discovery.trials(..., for_analysis=True)`. The headline
numbers barely moved, which is itself reassuring.

### Flagged for investigation

- **NTS01 has no usable LH or SKH data at all.** All 5 trials are
  `quality_omit`, so it produces no LH/SKH analysis and gets no per-activity
  gain (it correctly falls back to its subject-wide value). Its LH/SKH sync
  needs diagnosis before that subject can contribute to any cross-activity
  comparison.
- NTS01's LH/SKH framework `trial_data.npz` files on disk were computed
  during the reprocessing run with the since-discredited gains. They are
  `quality_omit` and therefore never read by any analysis, so this is inert --
  noted rather than re-run.
- **Lateral hop's ~26% floor is the real target**, not gain precision.

## Files

- `fit_ideal_gains.py` -> `results/ideal_gains_per_trial.csv` (one row per
  trial/side: ideal gain, current gain, RMSE under each, RMS raw insole force
  for the exact transfer-cost formula).
