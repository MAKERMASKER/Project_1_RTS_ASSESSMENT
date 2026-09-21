# Experimentation area

Sanctioned scratch space for trying out a new technique or concept in
isolation -- an alternative GRF filtering approach, a new calibration idea,
a different synchronization heuristic -- without touching production
modules in `code/` or mixing experimental output into `results/`.

## Rules

- Nothing here is imported by production code (`code/*.py` outside this
  folder). An experiment is a standalone script or notebook.
- Write your own output under `expt_temp/<your_experiment_name>/` -- never
  into `results/`.
- This folder is not covered by `code/tests/`.

## Promotion path

An experiment that proves out gets **rewritten** as a real module with
tests -- it is never copy-pasted from here into production. Once promoted,
delete the experiment folder (or leave it with a note pointing at the real
module) rather than maintaining two copies.

This replaces the role the old repo's `NONOOP/temp/` folder ended up playing
(744 undocumented one-off investigation scripts accumulated over time) --
same purpose, but an intentional, known location instead of accidental
clutter.
