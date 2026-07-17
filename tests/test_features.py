from __future__ import annotations

import pandas as pd

from src.features import FeatureEngineer


def test_feature_engineering_adds_expected_columns() -> None:
    df = pd.DataFrame(
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
    engineered = FeatureEngineer().engineer(df)

    assert "pulse_pressure" in engineered.columns
    assert "mean_arterial_pressure" in engineered.columns
    assert "age_cholesterol_interaction" in engineered.columns
    assert "systolic_bp_squared" in engineered.columns
    assert "bmi_category" in engineered.columns
    assert "bp_category" in engineered.columns
    assert "age_group" in engineered.columns
    assert "lifestyle_risk_score" in engineered.columns
    assert "alcohol_missing" in engineered.columns
    assert "active_missing" in engineered.columns
    assert "bp_bmi_interaction" in engineered.columns
    assert "age_bp_interaction" in engineered.columns
    assert engineered.loc[0, "pulse_pressure"] == 55.0
    assert engineered.loc[0, "bmi_category"] == 2
    assert engineered.loc[0, "bp_category"] == 3
    assert engineered.loc[0, "age_group"] == 1
    assert engineered.loc[0, "lifestyle_risk_score"] == 2
    assert engineered.loc[0, "alcohol_missing"] == 0
    assert engineered.loc[0, "active_missing"] == 0
