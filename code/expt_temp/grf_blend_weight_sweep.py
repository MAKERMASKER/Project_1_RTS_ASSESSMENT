"""Experiment: does the insole/COM blend weight change landing-peak-GRF agreement?

Motivated by resources/literature_notes.md: COM-double-differentiation GRF
is known to badly underestimate landing peak force, while insole force alone
underestimates peaks more mildly. framework_processor.py's
insole_vertical_weight/com_vertical_weight blend is a plausible mitigation,
but its weight was never tuned for this project -- this sweeps it and
reports peak-vertical-GRF bias against the marker/force-plate ground truth,
for AB08's already-processed SLJ trials.

Writes only into expt_temp/grf_blend_weight_sweep/ -- never into results/.
Requires the marker pipeline to have already been run for AB08 (it re-runs
only the framework half, at each weight, against the existing marker/ID
output).

Run from code/: ..\opensimPy\Scripts\python.exe expt_temp\grf_blend_weight_sweep.py
"""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from activities import activity_for
from config import DEFAULT
from discovery import TrialDiscovery
from framework_processor import FrameworkProcessor
from outcomes import OutcomeExtractor
from subject import Subject

OUT_DIR = Path(__file__).resolve().parent / "grf_blend_weight_sweep"
INSOLE_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _rules_with_weight(base_rules: dict, insole_weight: float) -> dict:
    rules = dict(base_rules)
    rules["insole_vertical_weight"] = insole_weight
    rules["com_vertical_weight"] = 1.0 - insole_weight
    return rules


class _FixedRulesSubject(Subject):
    """A Subject whose .rules is pinned to a specific dict for this sweep,
    instead of what's in subject_rules/<ID>.json -- so the sweep doesn't
    require editing that file for every weight tried."""

    def __init__(self, subject_id, config, fixed_rules):
        super().__init__(subject_id, config)
        self._fixed_rules = fixed_rules

    @property
    def rules(self):
        return self._fixed_rules


def main():
    subject_real = Subject("AB08", DEFAULT)
    activity = activity_for("single_leg_jump")
    discovery = TrialDiscovery(DEFAULT)
    trials = discovery.trials(subject_real, activity, "all")
    print(f"Sweeping {len(trials)} AB08 SLJ trials x {len(INSOLE_WEIGHTS)} weights")

    framework_processor = FrameworkProcessor(DEFAULT)
    outcome_extractor = OutcomeExtractor(DEFAULT)
    base_rules = subject_real.rules

    rows = []
    for insole_weight in INSOLE_WEIGHTS:
        rules = _rules_with_weight(base_rules, insole_weight)
        subject = _FixedRulesSubject("AB08", DEFAULT, rules)
        for trial in trials:
            out_dir = OUT_DIR / f"weight_{insole_weight:.2f}" / trial.name
            experiment_trial = replace(trial, output=out_dir)
            try:
                framework_processor.process(subject, activity, experiment_trial)
                result = outcome_extractor.analyze_trial(subject, activity, experiment_trial, make_plot=False)
            except Exception as error:  # noqa: BLE001 -- this is an investigation script, log and continue
                print(f"  {trial.name} @ weight={insole_weight}: FAILED ({error})")
                continue
            if result is None:
                continue
            row, _series = result
            record = dict(zip(
                ["Trial", "Activity", "Condition", "Side", "PeakFrameworkGRF_N", "PeakMarkerGRF_N"],
                row[:6],
            ))
            record["insole_vertical_weight"] = insole_weight
            rows.append(record)

    summary = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "peak_grf_by_weight.csv", index=False)

    summary["bias_pct"] = 100 * (summary["PeakFrameworkGRF_N"] - summary["PeakMarkerGRF_N"]) / summary["PeakMarkerGRF_N"]
    agreement = summary.groupby("insole_vertical_weight")["bias_pct"].agg(["mean", "std", "count"])
    agreement.to_csv(OUT_DIR / "peak_grf_bias_by_weight.csv")
    print(agreement.to_string())


if __name__ == "__main__":
    main()
