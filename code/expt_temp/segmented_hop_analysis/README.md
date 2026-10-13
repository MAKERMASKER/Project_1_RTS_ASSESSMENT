# Segmented (per-cycle) analysis for lateral hop and skater hop

## Why this exists

LH and SKH are multi-cycle activities, but production framework processing
analyses only ONE canonical event per trial (`LateralHop`'s own docstring:
"Only this first single-support landing is analyzed"). Measured: LH/SKH
framework windows are **0.8-1.3 s with 1-2 contact bouts**, against real bouts
containing 6-52 cycles. SLJ's one-row-per-trial model therefore cannot express
these activities, and per-cycle segmentation is impossible from existing
framework output.

This widens the window to the whole bout and segments it into cycles. It does
**not** modify production sync: it calls `activity.synchronize()` to get the
validated shift (for LH, the user's own manual per-trial click from
`lh_manual_sync`), then rebuilds the window as the full insole<->OpenCap
overlap. Same anchor, wider window.

## Cycle definitions (as specified)

- **SKH**: left -> right -> left is ONE cycle (push-off left, land right,
  push-off right, land left). Consecutive cycles share their middle landing.
  Each cycle contributes TWO stance phases, one per side -- which is what makes
  an L-vs-R LSI possible for a bilateral activity.
- **LH**: one cycle is push-off -> land on the SAME leg (contact[k] through
  contact[k+1]). ACCEPTED only if successive foot landings are separated by
  **> 0.40 m mediolaterally**, from OpenCap marker data.

ML direction is the foot's own anatomical ML axis (`InsoleGeometry.frames()`'s
middle column, averaged over the two landings), not a global axis -- global
X/Z are not comparable between the marker lab and OpenCap (CLAUDE.md), whereas
the foot frame is rebuilt from that trial's own markers and is frame-invariant.

## Headline finding: the earlier LH/SKH error figures were inflated by bad sync

SLJ aligns marker to framework by matching the single GRF peak. **That breaks
completely on a multi-cycle bout** -- with ~30-50 near-identical peaks there is
nothing to lock onto. Cross-correlation of the whole vertical-force trace is
used instead, and both offsets are recorded per trial so their disagreement is
a sync-quality flag.

Peak-matching vs cross-correlation disagree by **> 1 s in 28 of 56 trials**,
worst case **16.4 s** (`AB07_EXO_SKH_1`). The plots confirm cross-correlation
is the correct one -- framework and marker track hop-for-hop across whole bouts.

Consequence, with correct sync and per-cycle windows:

| Activity | GRF RMSE, per-cycle (this) | GRF RMSE, whole-trial peak-aligned (earlier) |
|---|---|---|
| lateral hop | **163 N** | 454-470 N |
| skater hop | **155 N** | 233-243 N |

**The "lateral hop has ~26% error even with a perfect gain" conclusion from
the cross-activity gain check does not survive.** That figure came from
`outcomes.py`'s peak-aligned whole-trial comparison, which we now know
misaligns LH trials by up to 16 s. LH is far better than it appeared; the
error was mostly a windowing/alignment artifact, not physics. The
cross-activity gain conclusion itself (SLJ gain transfers, ~16 N cost) is
unaffected -- that fit used its own per-trial peak-of-own-side alignment over
contact frames -- but the *floor* it was compared against was wrong.

## Results (2026-09-29; 56 trials, 6 subjects, 1794 stance rows)

Cycles detected / accepted:

| Subject | LH cycles | LH accepted | SKH cycles | SKH accepted |
|---|---|---|---|---|
| AB03 | 180 | 165 (92%) | 40 | 40 |
| AB05 | 314 | 290 (92%) | 47 | 47 |
| AB07 | 308 | **7 (2%)** | 52 | 52 |
| AB08 | 240 | 148 (62%) | 32 | 32 |
| AB10 | 227 | 217 (96%) | 42 | 42 |
| NTS01 | 82 | 31 (38%) | 12 | 12 |

`EXO`/`L` is 0 everywhere for LH, as it must be -- EXO trials are right-leg
only by protocol (CLAUDE.md). A useful consistency check that the
condition/side bookkeeping is right.

### AB07 fails the 40 cm gate on a genuine technique difference

AB07's LH median ML landing separation is **0.263 m**, against 0.60-0.66 m for
AB03/AB05/AB10 and 0.42 m for AB08. Its maximum over 308 cycles is 0.502 m.

This is not a data fault: AB07's sync and segmentation are excellent (see
`plots/lateral_hop/AB07/`), with framework and marker overlapping almost
exactly. AB07 simply hopped **fast and short** -- 52 hops in 30 s (1.7/s)
versus AB08's 30 in 26 s (1.15/s). A different movement strategy, not a
different measurement quality.

**Consequence: AB07 has no usable LH data under the 40 cm rule, so its LH LSI
is undefined (0 accepted NOEXO-L stances).** Whether that is the correct
scientific outcome -- AB07 did not perform the task as specified -- or whether
the gate needs to be subject-relative, is a study-design decision, not a
technical one. Flagged rather than silently worked around.

### GRF accuracy over accepted stances

| Activity | Fy RMSE | Fx RMSE | Fz RMSE |
|---|---|---|---|
| lateral hop | 163 N | 88 N | 65 N |
| skater hop | 155 N | 119 N | 58 N |

Marker Fx/Fz are reprojected into OpenCap's frame before comparison
(`opencap_X = +marker_Z`, `opencap_Z = -marker_X`), exactly as `outcomes.py`
does. Fy needs no correction.

## Not in this pass (deliberate)

Joint moments, support moment, work and COP all need the inverse-dynamics
stage re-run over the widened whole-bout window. That is deferred until the
segmentation above is confirmed, since ID over 30 s bouts is expensive and
would be wasted if the cycle definition changes. Their LSI rows are therefore
NaN rather than silently dropped.

## Files

- `segment_hops.py` -> `results/cycle_metrics.csv`, `results/plots/<activity>/<subject>/<trial>_cycles.png`
  (whole-bout framework-vs-marker GRF with cycles drawn on; green = accepted,
  red = rejected -- this is the plot for judging segmentation and sync)
- `summarise_cycles.py` -> `results/cycle_counts.csv`, `results/count_symmetry.csv`,
  `results/lsi_summary.csv`, `results/sync_quality.csv`, `results/plots/summary_counts.png`

Output is kept under `expt_temp/` rather than `results/<subject>/analysis/`
pending confirmation of the cycle definitions -- see
[[feedback-activity-analysis-authorization]].
