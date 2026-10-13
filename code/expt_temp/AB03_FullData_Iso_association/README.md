# AB03 -- SLJ vs isokinetic LSI, by phase and contraction mode

Uses `Isokinetic_Study_AB02_to_AB09_VALIDATED.xlsx`, which adds hamstring
(flexion) alongside quadriceps (extension). AB03 is the only subject with a
complete isokinetic set across all three comparisons (63 LSI rows); the others
have NaN for EXO power and H:Q because their with-device reports are short-form
and the workbook's audit deliberately removed derived ratios rather than
fabricate them.

Movement side: 15 SLJ trials (5 NOEXO-L, 5 NOEXO-R, 5 EXO-R). The 16th,
`AB03_NOEXO_SLJ_L_2`, has no OpenCap trial -- a documented gap.

## How SLJ and isokinetic map onto "flexion" and "extension"

"Flexion" names three separable things: the direction of joint MOTION, the
direction of the MOMENT, and the MUSCLE group. Isokinetic Con/Con aligns all
three by design. SLJ does not.

| | motion | moment | muscle | contraction |
|---|---|---|---|---|
| Isokinetic extension | extends | extensor | quadriceps | concentric |
| Isokinetic flexion | flexes | flexor | hamstrings | concentric |
| SLJ push-off | extends | extensor | quadriceps | concentric |
| SLJ landing | **flexes** | **extensor** | quadriceps | **eccentric** |

Verified on AB03, not assumed: through push-off `knee_angle_r` falls 68.8 ->
4.1 deg (knee straightening) while the moment stays negative and power is
positive -- concentric quadriceps, so negative == extensor. At landing the knee
flexes while the moment stays negative and power goes negative -- eccentric
quadriceps. **Every joint carries an extensor moment in both phases**; only the
power sign flips:

| | push-off (0.27 s) | landing (0.78 s) |
|---|---|---|
| support moment | +298.8 mean | +238.8 mean |
| hip | -82.5 Nm, **+224 W** | -135.7 Nm, **-81 W** |
| knee | -62.6 Nm, **+186 W** | -32.2 Nm, **-67 W** |
| ankle | -153.7 Nm, **+489 W** | -70.8 Nm, **-25 W** |

That is why an extensor/flexor split of the knee moment fails: the flexor bin
catches only noise (0-85 Nm against a 165-336 Nm extensor peak), giving LSIs
with 281-654 point CIs. Replacing it with a phase/mode split brings the median
CI to **39 points**.

Isokinetic flexion's honest counterpart is the hamstrings' *other* action --
biarticular **hip extension**, concentric, during push-off.

## Result 1: the specificity test is UNDERPOWERED, not negative

Every SLJ metric was compared against BOTH isokinetic groups, so "does SLJ
extension track isokinetic extension specifically" could be measured rather
than assumed.

| | predicted group closer | mean margin |
|---|---|---|
| Framework | 11/25 rows | -2.7 points |
| Marker | 13/25 rows | -0.2 points |

Both are coin-flips. But that does **not** show the association is
non-specific, because the test cannot discriminate here:

- gap between the two isokinetic targets: median **5.8** LSI points (0.9-15.2)
- width of the movement LSI 95% CI: median **39.0** LSI points (14.0-77.9)

**The targets sit 6.7x closer together than the movement uncertainty.** No
result was obtainable either way.

The root cause is subject selection: **AB03 is essentially symmetric.** Every
isokinetic LSI lies between 91.9% and 108.4% -- within about +/-9% of perfect
symmetry, for both extension and flexion. With both targets near 100 and barely
separated, there is nothing for the movement side to discriminate.

**Specificity can only be tested on subjects with genuinely divergent extension
vs flexion asymmetry.** That is a subject-selection requirement, not something
more trials on AB03 can fix.

## Result 2: mode matching came out backwards

Mean |SLJ - predicted isokinetic|:

| | mode-matched (push-off, concentric) | mode-MISmatched (landing, eccentric) |
|---|---|---|
| Framework | 21.1 | **18.1** |
| Marker | 35.6 | **19.3** |

Mechanism predicts push-off should win, since the reference is concentric.
Landing wins instead, for both pipelines.

Before reading that as a real effect, note push-off is the noisier measurement:
it is short (0.27-0.32 s vs 0.50-0.83 s), carries small work values (knee
40-96 J), and runs furthest from the reference speed -- **249 deg/s mean and
571 deg/s peak against the isokinetic 60 deg/s**, a 4-9x mismatch where the
force-velocity relationship bites hardest. Landing averages 136 deg/s.

So this is at least as likely to be push-off measurement noise as a genuine
eccentric advantage. It is not resolvable from one subject.

## Result 3: a marker-pipeline artifact worth fixing

`Marker_PushOff_AnkleConcWork_J` gives LSI **319.8%** (NOEXO_L_vs_R) and
**340.0%** (EXO_R_vs_NOEXO_R) where the framework gives 120.0% and 103.9%.
A 3x LSI is not physiological. Marker ankle concentric work over the short
push-off window looks unreliable -- likely ankle-angle differentiation over a
brief, fast window. Flagged, not used.

## What holds up

- **Phase/mode decomposition is the right structure.** Median CI 39 points
  versus 242-934 for anything built on knee flexor moment.
- **Hip extensor is the usable flexion route.** Stable (249-518 Nm), unlike
  knee flexor (0-85 Nm).
- **Support moment behaves as expected** -- its CIs are the tightest of the set
  (e.g. push-off 81.9-95.9), consistent with inter-joint compensation averaging
  out.
- **Landing knee extensor peak overstates asymmetry**: framework 129.7%
  [107.6, 157.0], CI excluding 100, against isokinetic 98.4%. Work and impulse
  agree far better. This reproduces the peak-vs-impulse pattern already in
  `resources/literature_notes.md` from an independent direction.

## What this does not establish

One subject. No correlation is computed and none is meaningful. CIs are over
trials, so they capture within-subject variability only -- nothing about
whether AB03 is representative.

## Recommendations

1. **Add `EXO_R_vs_NOEXO_R`** to `config.lsi_definitions` -- within-limb,
   removes anatomical variance, already supplied by the workbook.
2. **Use phase/mode metrics, not extensor/flexor bins**, for jump activities.
3. **Route flexion through hip extensor**; do not report knee-flexor or
   knee-derived H:Q LSI for landings.
4. **Prefer work/impulse over peak.**
5. **Carry bootstrap CIs into the LSI output** -- without them the underpowered
   specificity test above would have read as a clean negative result.
6. **To test specificity at all, select subjects whose isokinetic extension and
   flexion LSIs actually diverge.** Symmetric subjects cannot answer it.
7. Data collection: the reference is concentric-only and single-speed (60 deg/s).
   An eccentric protocol would let landing be mode-matched; multiple speeds
   would let the 4-9x velocity mismatch be characterised instead of assumed to
   cancel in the ratio.

## Files

- `extract_phases.py` -> `results/ab03_phase_kinetics.csv` (per trial, per
  phase: extensor peak/impulse, concentric/eccentric work, mean velocity at
  hip/knee/ankle, plus support moment; framework and marker)
- `associate_phases.py` -> `results/ab03_phase_association.csv`,
  `results/plots/ab03_phase_association.png`
- `extract_moments.py` / `associate.py` -- the earlier extensor-vs-flexor
  split, superseded. Kept because it is the evidence for why that
  decomposition was abandoned.
