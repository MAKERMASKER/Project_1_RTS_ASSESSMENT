# Fy gain trial screening -- which trials can support a vertical-gain estimate?

## The question

The COM+H-dot Fy estimator (`../collect_insole_com_rot_fy_visualization.py`,
plots in `validation/sensor_insole/_RAWInsole_COM_ROT_Fplate_Fy/`) is clean on
some trials (e.g. `AB05_NOEXO_SLJ_L_5`) and unusable on others (e.g.
`AB03_EXO_SLJ_R_6`, spikes to ~2900 N). **Outside the lab there is no force
plate to tell which is which.** Running the estimator and eyeballing the
result is not a feasible field workflow, so we need a screening rule
computable from the kinematics of the loading alone.

This check computes candidate ground-truth-free predictors AND the real
ground-truth outcome for every trial, so we can use lab ground truth to
*validate a rule that will not need ground truth*.

Note this is a **uniform** rule in the sense the project requires: one
criterion applied identically to every trial and subject. Which trials pass
varies, but the decision procedure never depends on knowing in advance who is
well-calibrated.

## The mechanism being screened for

The Fy term inverts the sagittal-plane moment balance:

```
M_z = lever_x * F_y - lever_y * F_x
  =>  F_y = (M_z + lever_y * F_x) / lever_x
```

`lever = cop - com`, so `lever_x` is the anteroposterior COM-to-COP offset.

- Error in `M_z` is amplified by `1/|lever_x|`.
- Error in `F_x` -- already the noisiest channel (video-derived horizontal COM
  acceleration) -- is amplified by `|lever_y|/|lever_x|`.

At the current mask floor (`MIN_AP_LEVER_M = 0.02 m`) with a typical
`lever_y ~ 0.9 m`, that is a **~45x amplification onto the noisiest input**,
which is exactly the spiking seen in the bad trials.

Physically: a trial carries information about `F_y` only when the subject
spends meaningful contact time with the COM offset fore/aft of the COP, so
that the `lever_x * F_y` term is a substantial **share** of the sagittal
moment rather than a small residual between two large terms. In a single-leg
jump the COM *must* pass over the foot at some point (`lever_x = 0` is
guaranteed), so the discriminator is dwell time away from zero, not whether
zero is ever crossed.

## Why insole COP is legitimate input (no circularity)

COP is a pressure **centroid** -- invariant to the overall force gain being
calibrated. Scaling every pressure cell by a constant leaves the centroid
unchanged. So a screening rule may use OpenCap kinematics + insole COP and
still be honestly independent of the gain it is screening for.

**Caveat**: the per-subject rigid COP offset (`rigid_offset_local_m` from
`InsoleCalibration`) is a lab-fit geometric constant. It is not a force-gain
quantity, but it does mean true field deployment needs that offset from
somewhere (a one-time geometric measurement, or an assumed default). Flagged
honestly rather than glossed.

## The outcome variable, and the trap it avoids

A trial can post a low whole-trace RMSE **for the wrong reason**: where the
lever is unsafe, the blend falls back to pure COM-translational, which at rest
simply equals body weight and therefore matches the plate. That is the H-dot
term being *bypassed*, not working. (This is very likely what makes
`AB05_NOEXO_SLJ_L_5`'s quiet-stance segment sit on the plate so neatly.)

So the headline outcome is:

```
DeltaRMSE_active_N = RMSE(translational) - RMSE(blend)
```

computed **only over frames where the H-dot term is actually active**.
Positive => the H-dot term genuinely improved on pure translational there.
Whole-trace RMSE is also recorded, but is not the thing being predicted.

## Files

- `collect_metrics.py` -- one row per trial: kinematic predictors (lever-arm
  distribution, amplification factor, identifiability share, mask
  activity/crossings, contact bouts/duration) + ground-truth outcomes.
  Caches per-frame arrays to `results/cache/` so the analysis can be re-run
  without redoing the expensive OpenSim COM/momentum-rate computation.
- `analyze_screening.py` -- ranks predictors by Spearman correlation against
  the outcome, writes `results/plots/predictors_vs_outcome.png`, and scores
  simple threshold rules by precision/recall against the "did the H-dot term
  help" label.

## Results (2026-09-25, all 6 subjects, 84 trials)

**The screening question turns out to be moot: there is no subset of trials
with potential, because the AP lever arm is structurally too small in this
task for ANY trial.**

### The H-dot Fy term does not work, anywhere

| | value |
|---|---|
| trials where H-dot improved on pure translational | **2 / 84 (2%)** |
| median `DeltaRMSE_active_N` | **-311 N** (negative = H-dot made it worse) |
| median RMSE on active frames, translational vs blend | **269 N -> 554 N** |

Worse for **every subject**, without exception:

| Subject | median DeltaRMSE_active (N) | RMSE translational (N) | RMSE blend (N) |
|---|---|---|---|
| AB03 | -410 | 340 | 741 |
| AB05 | -278 | 307 | 553 |
| AB07 | -354 | 177 | 513 |
| AB08 | -183 | 266 | 490 |
| AB10 | -281 | 285 | 549 |
| NTS01 | -473 | 160 | 661 |

The two "helped" trials are `AB05_NOEXO_SLJ_L_5` (+0.6 N) and
`AB05_NOEXO_SLJ_R_2` (+18.0 N) -- both with `ActiveFraction` ~0.13-0.14,
i.e. the H-dot term was switched off for ~87% of contact. These are the term
being *absent*, not the term working. Notably `AB05_NOEXO_SLJ_L_5` is the
trial that looks best by eye in the visualization folder -- confirming the
trap this check was designed to catch.

### Why: the task's own balance requirement guarantees ill-conditioning

| Subject | median AP lever (m) | median amplification `lever_y/lever_x` |
|---|---|---|
| AB03 | 0.033 | 27x |
| AB05 | 0.013 | 70x |
| AB07 | 0.018 | 48x |
| AB08 | 0.013 | 61x |
| AB10 | 0.031 | 28x |
| NTS01 | 0.022 | 36x |

- **Only 1 of 84 trials** has any contact frame with AP lever above 15 cm.
- The **largest fraction of contact above 10 cm in any trial is 12.75%**;
  in most trials it is zero.
- COM height above COP is ~0.85-0.90 m throughout, so the geometry runs
  ~30-70:1 against solving for `F_y`.

This is not a data-quality problem and no window/threshold fixes it. In a
single-leg jump, balance *requires* the COM to stay over the base of support,
so the AP COM-to-COP offset is structurally a couple of centimetres while the
COM sits ~0.9 m up. Inverting the sagittal moment balance for the vertical
force is then equivalent to **weighing an object by measuring torque about a
point directly beneath it** -- the quantity you want has almost no moment arm,
while the noisiest input (`F_x`) has a huge one.

### No screening rule exists (and the correlations say why)

Every threshold rule tested scores precision 0.00-0.02, identical to the
2% base rate -- i.e. no better than keeping everything, because there is no
"good" class to select. All predictor correlations are weak (|rho| < 0.27),
and their *signs* tell a consistent story:

- `LeverX_abs_p75_m` rho = **-0.24** -- more lever => *worse* outcome
- `ActiveFraction` rho = **-0.24** -- more H-dot participation => *worse*
- `Amplification_p25` rho = **+0.25** -- *higher* amplification => *better*

All three say the same thing: the more the H-dot term actually engages, the
more damage it does. High amplification correlates with a *better* result only
because it makes the mask reject more frames, falling back to pure
translational. **The best available "screening rule" is to leave the term
off.**

### Implications

1. Do not extend the H-dot correction to Fy. Production is right to keep
   `rotation_force_*` on Fx/Fz only, where the divisor is the large vertical
   lever (~0.9 m, masked at 0.25 m) rather than the tiny AP one.
2. The `MIN_AP_LEVER_M = 0.02` floor used in the visualization is far too
   permissive (~45x amplification at that value) -- but no achievable floor
   rescues it, since raising it simply disables the term.
3. The vertical-gain calibration problem needs a route that does **not**
   depend on the AP moment arm. The flight-time / impulse-momentum idea
   already noted in `resources/opensimad_gain_calibration_investigation.md`
   is attractive precisely because it uses the vertical axis directly
   (takeoff velocity from flight time vs. the insole's own integrated
   impulse) and never inverts a moment balance.
