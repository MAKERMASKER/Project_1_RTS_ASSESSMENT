# State-machine insole<->OpenCap auto-sync

Tests whether the insole<->OpenCap offset can be recovered automatically from
a per-foot contact state machine plus event-train matching -- using **only**
insole and OpenCap. The force plate never participates: it is validation, not
pipeline.

Validated against the **70 hand-synced references** (`lh_manual_sync` /
`skh_manual_sync`), whose clicked pairs imply
`offset = opencap_click_s - insole_click_s`, exactly what `_finish_sync`
derives. Reproducing them means the manual syncing can be automated.

## Method

1. **Both modalities -> the same discrete per-foot contact signal.** Insole
   force is a direct contact measurement. OpenCap has no force, so contact is
   inferred from foot height near the ground AND low horizontal foot speed,
   with the ground taken per trial as a low percentile of foot height (so it
   self-calibrates and never depends on OpenCap's vertical origin).
2. **The same two-stage state machine on each** (FLIGHT <-> STANCE):
   hysteresis kills boundary chatter; a minimum dwell rejects glitches; a
   maximum stance duration reclassifies a long hold as STANDING, which never
   enters the event train -- structurally preventing the documented failure of
   mistaking a static pre-trial stance for the first hop. Durations are
   `LateralHop`'s own validated 0.15 s / 1.8 s, not new guesses.
3. **Histogram voting to match the trains.** Every (insole *i*, OpenCap *j*)
   pair votes for `s_j - t_i`; correct pairs agree, spurious ones scatter, so
   the true offset is the mode. Immune to unequal lead-in -- the insole records
   continuously while OpenCap is trimmed, which is what breaks first-event
   anchoring.
4. **Self-validation** via the spread of matched residuals, plus a linear-trend
   check for clock drift.

## Results (2026-09-29; 79 trial/side results, 6 subjects)

### It works, and well

| | |
|---|---|
| median abs error vs hand-sync | **0.071 s** |
| within 0.10 s | 58/70 |
| within 0.25 s | 66/70 |
| within 1.00 s | 68/70 |
| median residual spread | 0.016 s |

By activity: **skater hop is essentially solved** (39 references, median 0.080 s,
**worst case 0.172 s**). Lateral hop is median 0.062 s but has a tail -- see
below.

For context, the reference itself is a human clicking a plot, so its own
precision is probably +/-50-100 ms. The median error is at roughly the level of
the reference's own uncertainty.

### Drift is confirmed absent

Median |residual slope| **0.00074**, max 0.0047. The "one constant offset per
trial, no drift" model is now empirically verified rather than assumed -- and
the drift check costs nothing, since it falls out of the same matched residuals.

### NTS01 -- a subject currently with no usable LH/SKH data -- syncs confidently

NTS01 has no manual sync, so it falls through to the unreliable automatic
method, and **every one of its LH/SKH trials is `quality_omit`**. The state
machine returns confident offsets for all 7 of its trial/sides (spreads
0.002-0.037 s, comparable to the best hand-synced trials). Its three LH trials
land at -2.65 to -2.68 s.

This is the concrete practical win: a whole subject's LH/SKH data is currently
discarded for want of a sync that this method appears to supply.

## The failure mode: off-by-one-hop-period, and the spread CANNOT see it

All four LH failures are off by **0.53, 0.60, 1.00, 1.16 s** -- and the LH hop
period is ~0.6-1.0 s. They are not random errors. **They are off-by-one-hop
matches.**

An LH bout is 30-50 near-evenly-spaced hops, so the difference histogram has
several near-equal peaks one hop period apart. The method takes the tallest,
which is sometimes the neighbour.

The critical consequence: **the residual spread does not detect this.** An
off-by-one-period match is internally perfectly consistent -- every event pairs
with a neighbour just as well -- so the spread stays tight (0.017-0.042 s) while
the answer is a full second wrong. Median spread is 0.015 s for good matches
against 0.041 s for bad ones: a real signal, but far too weak to gate on.

`plots/lateral_hop/AB08/AB08_NOEXO_LH_R_1_R_states.png` shows this directly:
31 events aligned, error -0.084 s, and a train so regular that a one-period
shift would look almost as good.

Skater hop is immune because it has only 5-16 events with an alternating L/R
structure and irregular timing -- no periodic degeneracy to fall into.

### The fix is the duration fingerprint

Touchdown *times* alone cannot break a periodic degeneracy. Stance and flight
*durations* can: real hops are not identical, so the sequence of durations is a
fingerprint that matches at exactly one alignment. This is the same gap-matching
insight the TripleHop work already had to adopt; this experiment supplies direct
evidence that lateral hop needs it too, and shows precisely why (the failures
are all exactly one period out).

Not implemented here -- this pass was to establish whether the state-machine
approach is viable at all. It is: 66/70 within 0.25 s, SKH worst case 0.172 s,
with one well-characterised and clearly addressable failure mode.

## Follow-up: the foot frame is not the problem -- the OBSERVABLE is

`foot_frame_sync.py` rebuilt the OpenCap detector on the standard framework
foot anatomical frame (`InsoleGeometry.frames()`), projecting both ends of the
foot onto its own sole plane along the foot's OWN vertical axis (so the
geometry stays correct through heel-strike and toe-off) and taking the LOWEST
sole point. This fixes a genuine defect: `InsoleGeometry.foot_height()`
averages only `toe1`/`toe5` and ignores the heel entirely, so it reads airborne
at heel-strike and in-contact after heel-lift. (That function was written as an
SLJ toe-height sync diagnostic, not a contact detector -- a misuse, not a bug
in it.)

**It made almost no difference**: median error 0.070 s vs 0.071 s, 65/70 vs
66/70 within 0.25 s. Closer on 36/70, worse on 2, tied on 32 -- mildly more
often precise, but the ceiling did not move.

Plotting the CONTINUOUS signals (which the original binary-train plots hid)
shows why. **Height is a marginal observable for these hops.** The sole reaches
only **8-13 cm** in flight against a 3.5 cm enter / 7 cm exit band, so flight
barely clears the threshold and adjacent hops merge -- both height detectors
find ~14 events where the insole finds 16. Including the heel does not help,
because for these forefoot-dominant lateral hops the toes usually ARE the
lowest point.

**Foot pitch separates far better**: ~0-5 deg planted against 35-40 deg in
flight. In a lateral hop the foot rotates much more decisively than it rises.

| detector | median abs error | within 0.05 s | within 0.25 s | within 1.00 s | event gap vs insole |
|---|---|---|---|---|---|
| toe height | 0.071 s | 15/70 | 66/70 | 68/70 | -2 |
| foot frame (sole height) | 0.070 s | 15/70 | 65/70 | 67/70 | -2 |
| **foot pitch** | **0.028 s** | **48/70** | 61/70 | 64/70 | -4 |

Pitch is **2.5x more accurate and 3.2x more often within 50 ms**, but is
**worse on robustness** -- more gross failures and a larger event deficit.
That is the expected trade-off: pitch pins the touchdown *instant* sharply
because its dynamic range is large, but it cannot tell a foot lying flat ON
the ground from one momentarily flat IN the air, so it misses events and
occasionally mis-segments.

## Final pass: what actually works (`refined_sync.py`)

Scope correction from the user: the event-count deficit against the insole is
**expected and benign** -- OpenCap simply stops recording before the insole
does. Only the overlapping duration needs syncing, and the objective is
precision over that overlap. So the "-2 / -4 event gap" is not a defect and
robustness-vs-event-count is not the trade-off to optimise.

### The duration fingerprint does NOT work for lateral hop -- negative result

Proposed (confidently, in discussion) as the fix for the off-by-one-hop
degeneracy, on the strength of TripleHop's gap-matching precedent. Tested:
**it provides no benefit.**

| | pitch only | combined + plain match | combined + fingerprint |
|---|---|---|---|
| median abs error | **0.028 s** | 0.034 s | 0.033 s |
| within 0.05 s | **48/70** | 44/70 | 43/70 |
| within 1.00 s | 64/70 | **67/70** | 64/70 |

Why it fails: LH inter-touchdown intervals are regular to within ~38 ms median
against a 600-1000 ms hop period -- about 4-8%. Shifting the alignment by one
whole period therefore barely perturbs the interval sequence, so interval
agreement has almost no discriminating power. The hops are simply too
metronomic. TripleHop's three hops with distinct gaps are a different regime.

**A bug I introduced and fixed along the way**: the first version ranked
candidates on mean interval error alone, which let a candidate matching 4
events by luck beat the correct one matching 40 events with honest noise.
Max error was 10.3 s. Match count is what separates a gross miss; the
fingerprint can only ever be a tie-break between near-equal-count candidates.
Fixed (max error 2.6 s), but the approach still adds nothing.

### What DOES work: cross-method agreement as a confidence gate

The two detectors **fail on different trials** -- of 10 gross failures for
combined+plain, pitch-only is correct on 6, and vice versa. So their agreement
is a usable confidence signal, which is exactly what residual spread failed to
provide.

Accepting the pitch estimate only when the two methods agree within 0.05 s:

| | |
|---|---|
| accepted | **53 / 70 (76%)** |
| median abs error on accepted | **0.022 s** |
| p90 / max on accepted | 0.076 s / **0.178 s** |
| failures among accepted | **0 / 53** |
| flagged for review | 17, containing all 9 genuine failures |

**No failure slips through the gate.** 76% of trials sync fully automatically
at ~22 ms median precision -- at or below the precision of the hand-clicked
reference itself -- and the remaining 24% are flagged for the manual GUI that
already exists. That converts the sync from 100% manual to 24% manual, with
the automated portion provably clean on this validation set.

### Implication: combine them, do not choose

Height (with the speed gate) is the robust *detector* -- it knows the foot is
near the ground. Pitch is the precise *edge locator* -- it knows exactly when
the foot became flat. Detect stances on height/speed, then refine each
touchdown edge to the nearby pitch crossing. That should keep height's 66/70
robustness while gaining pitch's 48/70 sub-50 ms precision.

Not implemented. Note this is orthogonal to the off-by-one-hop matching
failure above: better edges will not fix a periodic degeneracy, and the
duration fingerprint will not fix soft edges. They are two independent
problems and both need doing.

## Files

- `foot_frame_sync.py` -> `results_foot_frame/foot_frame_vs_toe_height.csv`,
  `results_foot_frame/plots/<subject>/<trial>_<side>_footframe.png`
  (three stacked panels: sole height with thresholds, foot pitch + horizontal
  speed, and the three contact trains against the insole -- so a called
  contact can actually be diagnosed)
- `auto_sync.py` -> `results/auto_sync.csv`,
  `results/plots/<activity>/<subject>/<trial>_<side>_states.png`
  (insole vs OpenCap contact trains after alignment, titled with the auto
  offset, residual spread, match count and error against the hand-sync)
