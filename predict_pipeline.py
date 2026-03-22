"""Compatibility prediction script for the saved model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features import FeatureEngineer
from src.train import ModelTrainer
from src.utils import load_joblib


def main() -> None:
    """Run a sample prediction against the saved pipeline.

    Args:
        None.

    Returns:
        None.
    """
    pipeline = load_joblib(Path.cwd() / "models" / "pipeline.pkl")
    sample = pd.DataFrame(
        [
            {
                "age_years": 58,
                "gender_bin": 1,
                "systolic_bp": 145,
                "diastolic_bp": 90,
                "bmi": 29.5,
                "cholesterol_raw": 220,
                "glucose_raw": 120,
                "smoke": 1,
                "alcohol": 0,
                "active": 1,
                "target": 0,
                "source": "demo",
            }
        ]
    )
    engineered = FeatureEngineer().engineer(sample)
    trainer = ModelTrainer()
    probability = float(
        pipeline.predict_proba(engineered[trainer.NUMERICAL_FEATURES + trainer.CATEGORICAL_FEATURES])[0][1]
    )
    print({"risk_percent": round(probability * 100, 2)})


if __name__ == "__main__":
    main()
