"""Feature engineering module for the cardiovascular risk model."""

from __future__ import annotations

import pandas as pd

from src.logger import get_logger


LOGGER = get_logger(__name__)


class FeatureEngineer:
    """Create a small, explainable feature set for cardiovascular risk modeling."""

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add only the approved engineered features."""
        frame = df.copy()
        frame["pulse_pressure"] = frame["systolic_bp"] - frame["diastolic_bp"]
        frame["mean_arterial_pressure"] = frame["diastolic_bp"] + (frame["pulse_pressure"] / 3.0)
        frame["age_cholesterol_interaction"] = frame["age_years"] * frame["cholesterol_raw"] / 1000.0
        frame["systolic_bp_squared"] = frame["systolic_bp"] ** 2
        frame["bmi_category"] = pd.cut(
            frame["bmi"],
            bins=[0, 18.5, 25, 30, float("inf")],
            labels=[0, 1, 2, 3],
            include_lowest=True,
        ).astype("Int64")
        frame["bp_category"] = 0
        frame.loc[(frame["systolic_bp"] >= 120) & (frame["diastolic_bp"] < 80), "bp_category"] = 1
        frame.loc[(frame["systolic_bp"] >= 130) | (frame["diastolic_bp"] >= 80), "bp_category"] = 2
        frame.loc[(frame["systolic_bp"] >= 140) | (frame["diastolic_bp"] >= 90), "bp_category"] = 3
        frame["age_group"] = pd.cut(
            frame["age_years"],
            bins=[17, 39, 59, 90],
            labels=[0, 1, 2],
            include_lowest=True,
        ).astype("Int64")
        frame["lifestyle_risk_score"] = (
            frame["smoke"].fillna(0).astype(int) * 2
            + frame["alcohol"].fillna(0).astype(int)
            + (1 - frame["active"].fillna(1).astype(int))
        )
        frame["alcohol_missing"] = frame["alcohol"].isna().astype(int)
        frame["active_missing"] = frame["active"].isna().astype(int)
        frame["bp_bmi_interaction"] = frame["systolic_bp"] * frame["bmi"] / 100.0
        frame["age_bp_interaction"] = frame["age_years"] * frame["systolic_bp"] / 100.0
        LOGGER.info("Engineered feature set with %s columns", frame.shape[1])
        return frame

    def get_feature_names(self) -> dict[str, list[str]]:
        """Return feature groups used in ML and explainability."""
        numerical = [
            "age_years",
            "systolic_bp",
            "diastolic_bp",
            "bmi",
            "cholesterol_raw",
            "glucose_raw",
            "smoke",
            "alcohol",
            "active",
            "pulse_pressure",
            "mean_arterial_pressure",
            "age_cholesterol_interaction",
            "systolic_bp_squared",
            "lifestyle_risk_score",
            "alcohol_missing",
            "active_missing",
            "bp_bmi_interaction",
            "age_bp_interaction",
        ]
        categorical = ["gender_bin", "bmi_category", "bp_category", "age_group"]
        engineered = [
            "pulse_pressure",
            "mean_arterial_pressure",
            "age_cholesterol_interaction",
            "systolic_bp_squared",
            "bmi_category",
            "bp_category",
            "age_group",
            "lifestyle_risk_score",
            "alcohol_missing",
            "active_missing",
            "bp_bmi_interaction",
            "age_bp_interaction",
        ]
        return {
            "numerical": numerical,
            "categorical": categorical,
            "all_ml": numerical + categorical,
            "engineered": engineered,
        }


if __name__ == "__main__":
    demo_df = pd.DataFrame(
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
    engineered_demo = FeatureEngineer().engineer(demo_df)
    LOGGER.info("Feature engineer demo columns: %s", engineered_demo.columns.tolist())
