# Data-integrity audit -- all subjects, all activities

225 trials audited across AB03/AB05/AB07/AB08/AB10/NTS01 and
single_leg_jump / lateral_hop / skater_hop / triple_hop.

Trials are enumerated through `TrialDiscovery`, never by globbing the results
tree -- `results/<S>/framework/<activity>/` also holds stale `local_original`
and `local_repaired` directories from trc_repair, so a filesystem count
overstates the trial set (AB03 lateral_hop: 21 npz files, 7 real trials).

## PRIORITY 1 -- AB03 and AB05 subject masses are SWAPPED

`data/subject_metadata.csv` has the two subjects' masses transposed. Two
independent sources agree against it:

| | subject_metadata.csv | OpenCap sessionMetadata.yaml | isokinetic workbook |
|---|---|---|---|
| AB03 | **68.3 kg** | 89.0 kg (`subjectID: AB03`) | 200 lb = 90.7 kg |
| AB05 | **89.0 kg** | 68.3 kg (`subjectID: AB05`) | 147 lb = 66.7 kg |

68.3 and 89.0 appear in both rows, exactly transposed. The OpenCap models are
right; the CSV is wrong.

**This has split the two pipelines apart for these subjects.** The framework
uses OpenCap's own model untouched (`framework_processor.py`), while
`ModelScaler.prepare_lai_model()` rescales the MARKER model to
`subject.mass_kg`. Measured model masses:

| | marker model | framework model | discrepancy |
|---|---|---|---|
| AB03 | 68.3 kg | 89.0 kg | **30%** |
| AB05 | 89.0 kg | 68.3 kg | **23%** |
| AB07 | 64.4 | 64.4 | 0 |
| AB08 | 62.6 | 62.6 | 0 |
| AB10 | 72.0 | 72.0 | 0 |
| NTS01 | 82.0 | 81.1 | 1.1% |

So for AB03 and AB05 every framework-vs-marker **joint moment** comparison
compares models differing by 23-30% in mass. Inverse dynamics scales with
inertial properties, so the marker moments are systematically wrong relative
to the framework for exactly these two subjects.

### Actual blast radius -- narrower than it first looks

Corrected after checking which model each pipeline really loads (the first
pass grabbed whichever `.osim` sorted first and got this wrong):

- **Framework** loads OpenCap's own model (`trial.opencap_model`), whose mass
  is OpenCap's own estimate -- 89.0 for AB03, 68.3 for AB05, both matching the
  isokinetic workbook. **Framework GRF and moments are correct for both.**
- **Marker GRF** comes from the force plate, no model involved. **Correct.**
- **Only MARKER joint moments** use the rescaled model, so only they are
  affected.

And the effect there is **smaller than proportional**. Inverse-dynamics joint
moments during stance are dominated by `GRF x moment arm`; segment inertia and
weight are secondary terms. Scaling model mass by 23-30% therefore shifts the
moments by far less. Measured marker/framework peak-moment ratios:

| | mass ratio (mk/fw) | knee ratio | hip ratio |
|---|---|---|---|
| AB03 | 0.767 | 0.881 | 1.553 |
| AB05 | 1.303 | 1.015 | 1.377 |
| clean subjects | 1.000 | 0.877-1.330 (median 1.043) | 1.084-1.679 |

AB03's knee ratio is below the clean median, directionally consistent with a
light marker model, but **AB05 shows no elevation at all** and the clean
subjects' own spread (0.877-1.330) is wider than either effect.

So: the swap is a genuine data error that must be fixed, but its downstream
signature is currently **indistinguishable from ordinary inter-subject
pipeline disagreement**. That is precisely why it survived this long.

Note also: `subject_metadata.csv` and both OpenCap sessions give AB03 and AB05
the SAME height (1.753 m), while the isokinetic workbook says 1.829 m and
1.727 m. Heights should be checked too.

**Nothing downstream should be trusted for AB03/AB05 until this is resolved**,
including the AB03 isokinetic association work in
`expt_temp/AB03_FullData_Iso_association/` and
`expt_temp/EXO_device_signature/`.

## PRIORITY 2 -- SLJ is clean; the hop activities are not

| activity | n | peak ratio (fw/marker) | GRF RMSE % | correlation median | correlation min |
|---|---|---|---|---|---|
| **single_leg_jump** | 84 | **1.05** | **6.2%** | **0.930** | **0.847** |
| lateral_hop | 33 | 0.77 | 26.3% | **0.455** | **-0.025** |
| skater_hop | 21 | 0.78 | 15.7% | 0.741 | 0.077 |
| triple_hop | 3 | 0.74 | 19.3% | 0.870 | 0.102 |

**Not one SLJ trial carries a HIGH-severity flag.** Every one of the 36
flagged trials is lateral hop, skater hop or triple hop.

Two distinct problems in the hops:

1. **Systematic underestimation.** Peak ratio medians of 0.77-0.78 mean the
   framework reports ~23% less peak force than the plate across nearly all hop
   trials. Systematic, not random.
2. **Loss of waveform correspondence.** Lateral hop's median framework-vs-plate
   correlation is 0.455, against 0.930 for SLJ. Half of all LH trials have
   essentially no shape agreement.

## The 36 flagged trials

Worst first by framework-vs-plate correlation. `BW` = body weights.

| Subject | Activity | Trial | fw BW | mk BW | ratio | RMSE% | corr |
|---|---|---|---|---|---|---|---|
| AB03 | LH | AB03_EXO_LH_R_1 | 0.86 | 2.67 | 0.32 | 42.5 | **-0.03** |
| AB10 | LH | AB10_NOEXO_LH_L_2 | 2.18 | 2.71 | 0.80 | 26.3 | 0.04 |
| NTS01 | LH | NTS01_NOEXO_LH_L_1 | 1.93 | 2.06 | 0.94 | 34.3 | 0.07 |
| AB03 | SKH | AB03_EXO_SKH_1 | 0.96 | 2.15 | 0.45 | 30.4 | 0.08 |
| AB08 | LH | AB08_EXO_LH_R_2 | 0.99 | 2.97 | 0.33 | 47.3 | 0.08 |
| NTS01 | SKH | NTS01_NOEXO_SKH_1 | 0.93 | 1.74 | 0.54 | 36.8 | 0.08 |
| AB03 | LH | AB03_NOEXO_LH_L_1 | 1.03 | 3.13 | 0.33 | 27.6 | 0.08 |
| AB05 | LH | AB05_NOEXO_LH_L_2 | 2.40 | 3.25 | 0.74 | 34.6 | 0.09 |
| AB08 | TH | AB08_NOEXO_TH_R_1 | 2.00 | 2.77 | 0.72 | 19.3 | 0.10 |
| AB08 | LH | AB08_NOEXO_LH_L_1 | 1.93 | 2.73 | 0.71 | 29.0 | 0.11 |
| AB03 | LH | AB03_EXO_LH_R_3 | 0.98 | 2.25 | 0.44 | 28.4 | 0.15 |
| AB07 | LH | AB07_NOEXO_LH_R_3 | 2.33 | 2.84 | 0.82 | 33.2 | 0.20 |
| AB10 | LH | AB10_EXO_LH_R_1 | 1.74 | 2.75 | 0.64 | 39.8 | 0.25 |
| AB07 | LH | AB07_EXO_LH_R_2 | 2.16 | 2.81 | 0.77 | 33.4 | 0.27 |
| AB03 | SKH | AB03_NOEXO_SKH_2 | 0.82 | 2.66 | 0.31 | 40.1 | 0.28 |
| AB08 | SKH | AB08_NOEXO_SKH_1 | 1.24 | 2.07 | 0.60 | 28.4 | 0.29 |
| AB08 | SKH | AB08_NOEXO_SKH_2 | 1.23 | 2.26 | 0.54 | 26.9 | 0.31 |
| AB05 | LH | AB05_EXO_LH_R_2 | 2.45 | 3.24 | 0.76 | 42.8 | 0.34 |
| AB10 | LH | AB10_EXO_LH_R_2 | 1.79 | 2.72 | 0.66 | 43.9 | 0.43 |
| AB05 | LH | AB05_EXO_LH_R_1 | 2.79 | 3.26 | 0.86 | 26.5 | 0.43 |
| AB05 | LH | AB05_NOEXO_LH_R_1 | 2.89 | 3.63 | 0.80 | 24.6 | 0.44 |
| AB08 | LH | AB08_NOEXO_LH_L_2 | **0.71** | 2.90 | **0.25** | 43.5 | 0.45 |
| AB10 | LH | AB10_NOEXO_LH_R_1 | 2.18 | 2.79 | 0.78 | 22.2 | 0.46 |
| AB03 | LH | AB03_NOEXO_LH_R_2 | 1.98 | 2.42 | 0.82 | 18.3 | 0.46 |
| NTS01 | LH | NTS01_NOEXO_LH_R_1 | 2.43 | 2.33 | 1.04 | 25.2 | 0.47 |
| AB05 | SKH | AB05_EXO_SKH_1 | 1.98 | 2.70 | 0.74 | 17.0 | 0.48 |
| AB07 | LH | AB07_NOEXO_LH_L_1 | 2.75 | 2.64 | 1.04 | 25.5 | 0.49 |
| AB03 | SKH | AB03_EXO_SKH_2 | 0.90 | 2.31 | 0.39 | 25.7 | 0.49 |
| AB03 | LH | AB03_NOEXO_LH_R_1 | 0.85 | 2.56 | 0.33 | 28.9 | 0.61 |
| AB03 | SKH | AB03_NOEXO_SKH_1 | 0.89 | 2.36 | 0.38 | 34.1 | 0.64 |
| AB08 | LH | AB08_NOEXO_LH_R_1 | 1.73 | 3.01 | 0.58 | 19.2 | 0.73 |
| AB10 | SKH | AB10_NOEXO_SKH_1 | 0.90 | 2.27 | 0.40 | 28.7 | 0.77 |
| AB03 | TH | AB03_NOEXO_TH_L_1 | 3.00 | 4.07 | 0.74 | 13.1 | 0.87 |
| AB10 | TH | AB10_NOEXO_TH_R_1 | **10.54** | 4.61 | **2.29** | **56.0** | 0.97 |
| AB08 | SKH | AB08_EXO_SKH_1 | 1.23 | 2.09 | 0.59 | 36.2 | n/a |
| AB10 | LH | AB10_NOEXO_LH_R_2 | **0.17** | 2.89 | **0.06** | **80.7** | n/a |

### Worst individual cases

- **`AB10_NOEXO_LH_R_2`** -- framework peak **0.17 BW** against a plate peak of
  2.89 BW (ratio 0.06, RMSE 81%). The framework is reporting essentially no
  force where the plate sees a full hop landing.
- **`AB10_NOEXO_TH_R_1`** -- framework peak **10.54 BW**, far beyond any
  plausible landing, and 2.3x the plate.
- **`AB03_EXO_LH_R_1`** -- correlation **-0.025**, i.e. no relationship at all
  between framework and plate vertical force, plus a marker knee moment of
  **970.6 Nm** (physiologically impossible).
- **`NTS01_NOEXO_LH_L_1`** -- marker knee moment **835.7 Nm**.

### AB03's lateral hop and skater hop look systematically wrong

Across AB03's flagged LH/SKH trials the framework peak is consistently
**0.82-1.03 BW** while the plate reads **2.15-3.13 BW** (ratios 0.31-0.45).
That consistency across many trials points at a processing fault for this
subject/activity -- the framework window landing on a lightly-loaded phase, or
a side/plate assignment error -- rather than per-trial noise.

## Fully explained, no action

- **`NO_FRAMEWORK_OUTPUT` (82)** -- 79 are triple_hop (never processed; its
  state detector is not finalised) and 3 are already-documented gaps
  (AB03_NOEXO_SLJ_L_2 has no OpenCap trial, one AB10 trial has no insole file,
  one NTS01 trial has no sync overlap).
- **`QUALITY_OMIT`** -- informational; the activity's own sync already flagged
  these.

## A false positive in this audit, found and fixed

The first run flagged `COP_DISCONTINUITY` on **131 trials**. That was the
audit's bug, not the data's: `framework_processor` ZEROES the COP outside
contact, so any step measured across a contact transition is a jump from
(0,0,0) to a real lab position -- about 1 m, and purely an artefact of the
zeroing. Median max-step was 1.02 m before the guard. The check now requires
both frames of a step to be in contact and erodes one frame either side of
every transition. Issue count fell 332 -> 201.

## Files

- `audit.py` -> `results/trial_audit.csv` (one row per trial, all metrics),
  `results/issues.csv` (one row per flagged issue, severity-ranked)
