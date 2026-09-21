# GRF blend-weight sweep -- findings (AB08, 12 SLJ trials, 2026-09-11)

## Result

Sweeping `insole_vertical_weight` from 0.0 (pure COM double-differentiation)
to 1.0 (pure insole force) against `com_vertical_weight = 1 - insole_vertical_weight`,
mean peak-vertical-GRF bias vs. the marker/force-plate ground truth:

| insole_vertical_weight | mean bias (%) | std (%) | n |
|---|---|---|---|
| 0.00 (pure COM) | -25.8 | 8.7 | 12 |
| 0.25 | -29.0 | 6.0 | 12 |
| 0.50 | -20.7 | 5.9 | 12 |
| 0.75 | -11.9 | 6.8 | 12 |
| 1.00 (pure insole) | **-3.1** | 8.2 | 12 |

Bias is `100 * (PeakFrameworkGRF - PeakMarkerGRF) / PeakMarkerGRF` -- negative
means the framework underestimates the marker/force-plate peak, consistent
with the published literature (see `resources/literature_notes.md`).

## Interpretation

For peak vertical GRF specifically, **pure insole force (weight=1.0) has the
lowest bias** (-3.1%), clearly better than pure COM (-25.8%) and better than
every blend in between -- the blend does **not** help for this metric on
this subject; it actually gets worse in the middle of the range (0.25-0.5)
before improving again toward pure insole. This is one plausible reading of
"insole and COM have different error profiles" (see literature notes): here,
insole's error profile is simply closer to zero for this metric, so blending
in COM only pulls the estimate further from ground truth.

AB08's actual subject_rules (no override -> default `insole_vertical_weight=1.0,
com_vertical_weight=0.0`) already matches the empirically best-performing
setting for peak GRF bias on this subject -- this was not tuned deliberately,
it's the ported default, but it happens to be right for this metric/subject.

## Caveats -- do not generalize from this alone

- **n=1 subject.** NTS01 (the only other subject with framework data today)
  was not included in this run; different subjects' insole sensor
  calibration/placement could shift this picture.
- **Peak GRF only.** The blend may behave differently for impulse, COP
  accuracy, or joint moment agreement -- this sweep did not check those.
- **Variance (std) doesn't improve monotonically** with weight -- the
  std at weight=1.0 (8.2%) is not the lowest (weight=0.5/0.75 are lower,
  5.9-6.8%), so "best mean bias" and "most consistent" aren't the same
  setting here.
- No statistical test was run (n=12 trials, not independent subjects) --
  this is a descriptive result to guide further investigation, not a
  conclusion to bake into a changed default without checking at least one
  more subject and at least one more outcome metric.

## Reproduce

```powershell
cd code
..\opensimPy\Scripts\python.exe expt_temp\grf_blend_weight_sweep.py
```

Writes only under `expt_temp/grf_blend_weight_sweep/` -- never touches `results/`.
