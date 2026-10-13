#!/usr/bin/env bash
# Clean reprocessing of SLJ / LH / SKH for every subject, with the
# per-activity ideal gains applied (apply_ideal_gains.py --apply first).
#
# Triple hop is deliberately NOT included: its state detector is not yet
# finalised, so its insole<->OpenCap sync cannot be trusted yet.
#
# `cli.py framework` runs framework processing AND the analysis stage
# (see cli.py: run_framework then run_analysis), so this covers both.
#
# Run from code/:
#   bash expt_temp/cross_activity_gain_check/reprocess_all.sh
set -u
PY="../opensimPy/Scripts/python.exe"
SUBJECTS="AB03 AB05 AB07 AB08 AB10 NTS01"
ACTIVITIES="single_leg_jump lateral_hop skater_hop"

for activity in $ACTIVITIES; do
  for subject in $SUBJECTS; do
    echo "=== $subject / $activity ==="
    "$PY" cli.py framework --subject "$subject" --activity "$activity" --condition all 2>&1 \
      | tail -25
    echo "exit=$? for $subject/$activity"
  done
done
echo "REPROCESS_ALL_DONE"
