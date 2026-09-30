# Literature notes

Standing reference for published work relevant to this project's methods.
Add to this as you or Claude find more papers -- don't let findings live
only in a chat transcript. One section per topic; newest addition at the
bottom of each section with the date it was added.

**See also**: `resources/opensimad_gain_calibration_investigation.md` --
this project's own experimental log (not published literature) for the
"OpenSimAD as a marker-ground-truth-free insole gain reference"
investigation, paused 2026-09-24.

## Insole pressure GRF accuracy

- **loadsol accuracy vs. force plates** (running): mean bias for ground contact
  time, impulse, peak force, time to peak all within 0.6-3.4%; 95% of
  differences < 12%. [PubMed](https://pubmed.ncbi.nlm.nih.gov/29842825/)
- **Insole3 validity** (walking/running): excellent vGRF agreement (ICC > 0.94
  for peaks, valleys, impulse), but insoles *underestimate* vGRF peaks
  (-3.7% to 0.9% bias) and overestimate impulse (4.2-5.6% bias). High loading
  rate is not well resolved, attributed to low sampling rate.
  [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8951440/)

_Added 2026-09-09, initial literature pass for the SLJ pipeline plan._

## Markerless mocap / OpenCap GRF accuracy

- **OpenCap vertical jump GRF in children**: phase durations and impulse agree
  well (r = 0.83-0.98) with force-plate ground truth, but **peak landing force
  bias exceeds 40% (severe underestimation), ICC only 0.29**. Attributed to
  (a) OpenCap's 240 Hz kinematic sample rate vs. the force plate's 1000 Hz --
  the brief, fast landing transient is under-resolved -- and (b) low-pass
  filtering of video-derived acceleration attenuating the high-frequency
  impact component. Authors recommend trusting OpenCap more for push-off /
  propulsive-phase metrics than for landing-peak force.
  [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13259095/)
- Related: [Jumping towards field-based GRF estimation with OpenCap](https://www.sciencedirect.com/science/article/pii/S0021929024001210),
  [Markerless mocap GRF predictions across movement tasks](https://pubmed.ncbi.nlm.nih.gov/38503062/)

**Implication for this project**: this pipeline's `insole_vertical_weight`/
`com_vertical_weight` blend (mixing the COM-double-differentiation force
estimate with the insole's own force fraction) is exactly the kind of
mitigation this literature implies is needed -- insole and COM-derived
estimates have different, complementary error profiles. Treat the blend
weight as an investigation target (sweep it, measure landing-peak-GRF
agreement against ground truth), not an assumed-correct constant.

_Added 2026-09-09, initial literature pass for the SLJ pipeline plan._

**Update, 2026-09-11 -- first sweep result (AB08, n=12 trials, 1 subject)**:
pure insole (`insole_vertical_weight=1.0`) gave the lowest peak-vertical-GRF
bias (-3.1%) of any weight tried (0.0 to 1.0 in steps of 0.25), clearly
better than pure COM (-25.8%) or any blend in between -- the blend did not
help for this metric on this subject. See
`code/expt_temp/grf_blend_weight_sweep/FINDINGS.md` for the full table and
caveats (n=1 subject, peak GRF only, not yet checked against a second
subject or a second outcome metric -- not a settled conclusion).

## Insole-to-marker COP calibration

- **In-shoe COP: indirect force-plate vs. direct insole measurement**: a
  rigid-transform calibration (probe-based, marker-tracked) achieves
  ~15±4 mm AP / 8.5±3 mm ML accuracy for cutting/running tasks. Explicitly
  notes the rigid-body/constant-transform assumption between foot and insole
  as a real error source -- the same assumption this project's
  `insole_geometry.py`/`calibration.py` makes.
  [ResearchGate](https://www.researchgate.net/publication/230781150_In-shoe_center_of_pressure_Indirect_force_plate_vs_direct_insole_measurement)

_Added 2026-09-09, initial literature pass for the SLJ pipeline plan._

## Double-integrating GRF for displacement (COM reconstruction)

Investigated 2026-09-23 while validating the framework pipeline's vertical
GRF (see `code/expt_temp/grf_double_integration_check/`): double-integrating
GRF-derived acceleration and comparing the reconstructed COM trajectory
against the real (kinematics-measured) one. Known, general issues with this
technique -- some confirmed empirically against this project's own data,
not just cited from elsewhere:

- **Quadratic drift from any constant bias.** A textbook signal-processing
  fact (the same reason inertial-navigation/IMU dead-reckoning drifts over
  time): a *constant* force-measurement bias of magnitude `e` produces a
  position error growing as `(e/mass) * t^2 / 2`. A bias small enough to
  look "reasonably calibrated" against peak-force comparisons (single-digit
  percent) can still produce large absolute displacement error once
  integrated twice over a ~1.5s window -- confirmed on this project's own
  best-calibrated subject (AB07, ~5% mean peak-GRF error) still showing
  ~950mm of vertical displacement drift.
- **Unknown initial velocity is unavoidable.** Displacement reconstruction
  needs a velocity boundary condition; there's no way to measure true
  initial velocity directly from GRF alone, so it must be estimated (here,
  finite-differenced from the first couple of measured COM samples), and
  any error there propagates linearly across the whole window on top of
  the bias-driven quadratic error -- a second, independent error source
  that can't be separated from GRF error using displacement alone.
- **Impact-transient under-resolution disproportionately corrupts double
  integration.** The already-documented OpenCap landing literature finding
  above (peak landing force bias >40%, from sample-rate mismatch + low-pass
  filtering killing the fast impact transient) matters *more* here than for
  a simple peak comparison: double integration concentrates its error where
  the signal changes fastest (push-off and landing), because that's where
  a shape/timing mismatch (not just a magnitude one) has the most impulse
  to get wrong. Confirmed directly on real data: AB07's reconstructed
  vertical velocity overshot the real takeoff velocity by ~1 m/s within
  0.3s of push-off, and failed to reproduce the real landing's sharp
  negative-velocity excursion at 0.6-0.75s.
- **Smoothing does not fix this -- tested, not assumed.** Low-pass
  filtering (4.5 Hz, this project's existing `id_lowpass_hz`) both the GRF
  signal and the kinematic reference before integrating was tested across
  all 6 subjects/all valid SLJ trials: it *worsened* the mean displacement
  error for 4 of 6 subjects (up to +41%) and left the overall mean
  essentially unchanged (1258mm -> 1267mm). This rules out high-frequency
  sensor noise as the dominant error source -- the mismatch is a genuine
  low-frequency shape/magnitude discrepancy at the timescale of the
  movement itself, which no amount of filtering can remove without also
  removing real signal.
- **Impulse (single integration) stays far more reliable than displacement
  (double integration).** Matches the existing OpenCap literature's own
  recommendation ("trust propulsive-phase/impulse metrics over landing-peak
  force") -- confirmed here too: the same subjects' impulse-momentum
  (velocity-change) errors were an order of magnitude more consistent
  (0.5-3.6 m/s across subjects) than their double-integrated displacement
  errors, and ranked the same subjects in the same good-to-bad order.
- **Insole-specific error profile differs from COM-derived error.** The
  Insole3 validity finding above (insoles *underestimate* vGRF peaks but
  *overestimate* impulse) means a "pure insole" vertical GRF (this
  project's preferred blend weight for peak-GRF accuracy, per the
  2026-09-11 sweep above) is not automatically the best choice for
  displacement-level accuracy -- the peak-accuracy-optimal setting and the
  displacement-accuracy-optimal setting could differ; not yet tested here.

**Practical takeaway for this project**: treat double-integration
displacement checks as a *relative*, amplified confirmation signal (good
for flagging which subject/trial is clearly worse than another -- it
caught AB10's broken calibration decisively) rather than as an absolute
accuracy metric in millimeters, and don't expect filtering to rescue a bad
result -- if smoothing doesn't help, the problem is in the signal's real
shape, not its noise floor.

## LSI / RTS clinical context

- LSI > 90% (single-leg hop distance) is a commonly used ACL-RTS cutoff, but
  hop-distance symmetry can mask persistent biomechanical asymmetry --
  vertical jump height/RSI symmetry is flagged as a *more* sensitive
  discriminator than horizontal hop distance.
  [SAGE systematic review](https://journals.sagepub.com/doi/10.1177/19417381231205267),
  [PMC: vertical jump symmetry vs biomechanics](https://pmc.ncbi.nlm.nih.gov/articles/PMC11195865/)

_Added 2026-09-09, initial literature pass for the SLJ pipeline plan._
