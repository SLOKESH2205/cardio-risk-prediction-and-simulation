from __future__ import annotations

import pandas as pd

from src.features import FeatureEngineer
from src.train import ModelTrainer


def test_model_trainer_knows_feature_columns() -> None:
    trainer = ModelTrainer(base_dir="d:/PROJECTS/VS CODE/heart_disease_prediction")
    feature_names = trainer.NUMERICAL_FEATURES + trainer.CATEGORICAL_FEATURES
    assert "age_years" in feature_names
    assert "mean_arterial_pressure" in feature_names


def test_feature_engineering_keeps_required_columns_for_modeling() -> None:
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
    model_features = ModelTrainer(base_dir="d:/PROJECTS/VS CODE/heart_disease_prediction").NUMERICAL_FEATURES + ModelTrainer(base_dir="d:/PROJECTS/VS CODE/heart_disease_prediction").CATEGORICAL_FEATURES
    assert all(column in engineered.columns for column in model_features)
