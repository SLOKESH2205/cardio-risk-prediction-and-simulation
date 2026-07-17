from __future__ import annotations

import pandas as pd

from src.simulation import apply_scenario, get_best_action, run_scenarios


def test_scenarios_return_expected_rows() -> None:
    base_df = pd.DataFrame(
        [
            {
                "age_years": 55,
                "gender_bin": 1,
                "systolic_bp": 145,
                "diastolic_bp": 90,
                "bmi": 29.1,
                "cholesterol_raw": 220,
                "glucose_raw": 120,
                "smoke": 1,
                "alcohol": 0,
                "active": 1,
                "target": 1,
                "source": "demo",
            }
        ]
    )
    modified = apply_scenario(base_df, "bp_minus_10")
    assert modified.iloc[0]["systolic_bp"] == 135.0


def test_best_action_works_for_improvements() -> None:
    df = pd.DataFrame(
        [
            {"scenario": "Original", "delta": 0.0},
            {"scenario": "BP -10", "delta": -5.0},
            {"scenario": "BP -20", "delta": -8.0},
        ]
    )
    best_action = get_best_action(df)
    assert best_action is not None
    assert best_action["scenario"] == "BP -20"
