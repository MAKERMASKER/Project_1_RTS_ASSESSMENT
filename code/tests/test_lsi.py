"""LsiCalculator.compute() must match a directly-computed groupby ratio for
every (comparison, metric) pair in config.lsi_definitions -- no hidden
hardcoding of the comparison axis or the arithmetic."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DEFAULT
from lsi import LsiCalculator, METRICS


def _synthetic_summary() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    metric_columns = sorted({column for _, _, framework_col, marker_col in METRICS for column in (framework_col, marker_col)})
    rows = []
    for condition, side, count in (("NOEXO", "L", 3), ("NOEXO", "R", 2), ("EXO", "R", 2)):
        for index in range(count):
            row = {
                "Trial": f"{condition}_{side}_{index}", "Activity": "single_leg_jump",
                "Condition": condition, "Side": side,
            }
            row.update({column: float(rng.uniform(50, 150)) for column in metric_columns})
            rows.append(row)
    return pd.DataFrame(rows)


def test_compute_matches_direct_groupby_ratio():
    summary = _synthetic_summary()
    ratios = LsiCalculator(DEFAULT).compute(summary)

    for name, test_condition, test_side, reference_condition, reference_side in DEFAULT.lsi_definitions:
        test = summary[(summary.Condition == test_condition) & (summary.Side == test_side)]
        reference = summary[(summary.Condition == reference_condition) & (summary.Side == reference_side)]
        for _category, metric, framework_col, marker_col in METRICS:
            expected_framework = 100 * test[framework_col].mean() / reference[framework_col].mean()
            expected_marker = 100 * test[marker_col].mean() / reference[marker_col].mean()

            row = ratios[(ratios.Comparison == name) & (ratios.Metric == metric)]
            assert len(row) == 1
            assert row["Framework_pct"].iloc[0] == pytest.approx(expected_framework)
            assert row["Marker_pct"].iloc[0] == pytest.approx(expected_marker)
            assert row["TestTrials"].iloc[0] == len(test)
            assert row["ReferenceTrials"].iloc[0] == len(reference)
