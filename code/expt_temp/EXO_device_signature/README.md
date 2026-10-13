# Can the SLJ pipelines detect the exoskeleton's direction-specific effect?

Extends `expt_temp/AB03_FullData_Iso_association/` from one subject to all
four that have both movement data and EXO isokinetic data: **AB03, AB05, AB07,
AB08**. (AB02/AB04/AB09 have isokinetic but no data folder; AB10/NTS01 have
movement but no isokinetic row.)

57 SLJ trials, all with a captured push-off contact.

## The signal being tested

`EXO_R_vs_NOEXO_R` compares the SAME limb with the device on and off, so
between-limb anatomy cancels. The isokinetic reference shows a consistent
directional effect -- the device raises knee EXTENSION and lowers knee
FLEXION -- in **every subject in the workbook**:

| AB02 | AB03 | AB04 | AB05 | AB07 | AB08 | AB09 |
|---|---|---|---|---|---|---|
| +13.7 | +12.7 | +24.1 | +18.5 | +12.5 | +40.1 | +41.0 |

(extension-metric mean minus flexion-metric mean, LSI points; positive in 7/7)

This is the only comparison carrying a large direction-specific signal.
`NOEXO_L_vs_R` averages just 2.8 points of deviation from symmetry versus 6.0
here, which is why the earlier specificity test on AB03's NOEXO data was
underpowered.

## Result 1: the test as first framed was MIS-SPECIFIED

The mechanism mapping routes hamstring/flexion through **hip** extensor
(hamstrings are biarticular; the knee flexor moment in a landing is noise) and
quadriceps/extension through **knee, ankle and support** moment. So grouping
SLJ metrics as "predicted Extension" vs "predicted Flexion" also groups them
as **knee/ankle vs hip**. The contrast measures inter-joint load distribution
at least as much as muscle-group specificity.

That confound produced a result that looked like a clean failure -- the
framework had the "wrong" sign in 0/4 subjects -- while actually measuring
something real but different. Reported here rather than quietly re-cut,
because the first framing is the one that would have been believed.

**Muscle-group specificity is therefore still untested.** Separating it would
need an SLJ metric that isolates hamstring from glute at the hip, which
inverse dynamics cannot provide (the hip extensor moment is both).

## Result 2: the device shifts load from knee to hip -- framework, 4/4 subjects

Re-cut by JOINT rather than by predicted muscle group (EXO vs NOEXO, same limb):

| Subject | Fw hip | Fw knee | Fw hip-knee | Mk hip | Mk knee | Mk hip-knee | isokinetic divergence |
|---|---|---|---|---|---|---|---|
| AB03 | 111.1 | 94.9 | +16.3 | 97.6 | 101.4 | -3.7 | 12.7 |
| AB05 | 87.5 | 64.8 | +22.8 | 89.8 | 91.4 | -1.5 | 18.5 |
| AB07 | 113.4 | 108.9 | +4.6 | 90.6 | 113.6 | -23.0 | 12.6 |
| AB08 | 134.3 | 89.0 | +45.3 | 145.2 | 90.9 | +54.4 | 40.2 |

Framework: **hip > knee in 4/4**, knee LSI below 100 in 3/4, hip above 100 in
3/4. Mechanistically coherent -- a device assisting knee extension lets the
knee do less, and demand moves proximally to the hip.

The marker pipeline does **not** reproduce this (hip > knee in only 1/4), which
is itself worth noting: the two pipelines disagree about where the device's
effect appears.

## Result 3: the correlation is NOT established -- it rests on one subject

The hip-minus-knee shift appears to track the isokinetic device effect:
framework r = +0.95 (p=0.050), marker r = +0.97 (p=0.028). Leave-one-out says
otherwise:

| dropped | framework r | marker r |
|---|---|---|
| none (n=4) | +0.95 (p=0.050) | +0.97 (p=0.028) |
| AB03 | +0.97 (p=0.165) | +1.00 (p=0.043) |
| AB05 | +0.96 (p=0.178) | +0.97 (p=0.152) |
| AB07 | +1.00 (p=0.009) | +0.99 (p=0.107) |
| **AB08** | **+0.78 (p=0.427)** | **+0.59 (p=0.598)** |

Removing AB08 collapses both. AB08's isokinetic divergence (+40.2) is more
than twice the next highest, and the remaining three span only 12.6-18.5 --
almost no range to correlate against. **This is a single influential point, not
a relationship.** It is a hypothesis worth testing on more subjects, nothing
more.

## Result 4: support moment is again the best-behaved metric

Mean |SLJ LSI - isokinetic group LSI| across the four subjects:

| metric | framework gap | fw rho | marker gap | mk rho |
|---|---|---|---|---|
| **Push-off support moment peak** | **8.3** | +0.8 | **6.2** | +0.4 |
| Landing support moment peak | 17.1 | -0.2 | 12.4 | +0.8 |
| Push-off hip extensor peak | 18.9 | -0.8 | 17.9 | -0.2 |
| Landing knee extensor peak | 21.7 | -0.8 | 14.2 | 0.0 |
| Push-off knee extensor peak | 28.5 | -0.4 | 25.7 | -0.8 |
| Landing hip eccentric work | 34.6 | -0.6 | 42.2 | -0.4 |
| Landing knee eccentric work | 36.0 | -0.8 | 27.8 | -1.0 |
| Push-off knee concentric work | 38.2 | -0.8 | 20.9 | +0.8 |
| Push-off hip concentric work | 43.0 | -0.4 | 33.1 | -0.4 |

Support moment peak is best for both pipelines, matching the AB03 finding.
Work metrics are consistently worst. The rho column is over 4 points and
should not be read as evidence of anything.

## Result 5: a marker-pipeline bug, confirmed across subjects

`Marker_PushOff_AnkleConcWork_J` gives LSI **340.0%** (AB03) and **389.0%**
(AB05) where the framework gives 103.9% and 97.7%. The AB03 artifact
generalises, so this is a systematic defect, not a one-trial glitch -- marker
ankle concentric work over the short, fast push-off window is unreliable,
most likely ankle-angle differentiation across that window. Excluded here and
flagged for fixing in `outcomes.py`'s work computation.

## What this does and does not show

Does: the isokinetic device effect is real and consistent (7/7 subjects); the
framework sees a consistent knee-to-hip load shift with the device (4/4); the
support moment is the most trustworthy movement metric; a marker work bug is
systematic.

Does not: establish that either pipeline detects the device's *direction-
specific* effect. The original specificity question remains unanswered because
the hip/knee proxy confounds it, and the one supporting correlation is a
single-point artifact.

## Files

- `extract_phases.py` -> `results/phase_kinetics.csv` (57 trials; per phase,
  per joint: extensor peak/impulse, concentric/eccentric work, mean velocity,
  support moment; framework and marker)
- `device_signature.py` -> `results/device_signature_summary.csv`,
  `results/device_signature_per_metric.csv`,
  `results/device_signature_per_metric_ranking.csv`,
  `results/plots/device_signature.png`
