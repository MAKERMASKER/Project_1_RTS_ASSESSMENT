# Literature notes

Standing reference for published work relevant to this project's methods.
Add to this as you or Claude find more papers -- don't let findings live
only in a chat transcript. One section per topic; newest addition at the
bottom of each section with the date it was added.

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

## LSI / RTS clinical context

- LSI > 90% (single-leg hop distance) is a commonly used ACL-RTS cutoff, but
  hop-distance symmetry can mask persistent biomechanical asymmetry --
  vertical jump height/RSI symmetry is flagged as a *more* sensitive
  discriminator than horizontal hop distance.
  [SAGE systematic review](https://journals.sagepub.com/doi/10.1177/19417381231205267),
  [PMC: vertical jump symmetry vs biomechanics](https://pmc.ncbi.nlm.nih.gov/articles/PMC11195865/)

_Added 2026-09-09, initial literature pass for the SLJ pipeline plan._
