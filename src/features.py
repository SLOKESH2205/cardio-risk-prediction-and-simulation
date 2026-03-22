"""Feature engineering module."""

from __future__ import annotations

import pandas as pd

from src.logger import get_logger


LOGGER = get_logger(__name__)


class FeatureEngineer:
    """Create domain-informed cardiovascular features."""

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer all required features.

        Args:
            df: Harmonized input dataframe.

        Returns:
            Feature-enriched dataframe.
        """
        frame = df.copy()
        frame["pulse_pressure"] = frame["systolic_bp"] - frame["diastolic_bp"]
        frame["bp_category"] = 0
        elevated_mask = frame["systolic_bp"].between(120, 129) & (frame["diastolic_bp"] < 80)
        stage_1_mask = frame["systolic_bp"].between(130, 139) | frame["diastolic_bp"].between(80, 89)
        stage_2_mask = (frame["systolic_bp"] >= 140) | (frame["diastolic_bp"] >= 90)
        frame.loc[elevated_mask, "bp_category"] = 1
        frame.loc[stage_1_mask, "bp_category"] = 2
        frame.loc[stage_2_mask, "bp_category"] = 3

        frame["age_group"] = 0
        frame.loc[(frame["age_years"] >= 40) & (frame["age_years"] < 60), "age_group"] = 1
        frame.loc[frame["age_years"] >= 60, "age_group"] = 2

        frame["bmi_category"] = 0
        frame.loc[(frame["bmi"] >= 18.5) & (frame["bmi"] < 25), "bmi_category"] = 1
        frame.loc[(frame["bmi"] >= 25) & (frame["bmi"] < 30), "bmi_category"] = 2
        frame.loc[frame["bmi"] >= 30, "bmi_category"] = 3

        frame["lifestyle_risk_score"] = (
            (frame["smoke"].fillna(0) * 2)
            + frame["alcohol"].fillna(0)
            + (1 - frame["active"].fillna(1))
        )
        frame["bp_bmi_interaction"] = frame["systolic_bp"] * frame["bmi"] / 100
        frame["age_bp_interaction"] = frame["age_years"] * frame["systolic_bp"] / 1000
        LOGGER.info("Engineered feature set with %s columns", frame.shape[1])
        return frame

    def get_feature_names(self) -> dict[str, list[str]]:
        """Return feature groups used in ML and explainability.

        Args:
            None.

        Returns:
            Dictionary of feature name groups.
        """
        numerical = [
            "age_years",
            "systolic_bp",
            "diastolic_bp",
            "bmi",
            "pulse_pressure",
            "cholesterol_raw",
            "glucose_raw",
            "lifestyle_risk_score",
            "bp_bmi_interaction",
            "age_bp_interaction",
        ]
        categorical = ["bp_category", "age_group", "bmi_category", "gender_bin"]
        engineered = [
            "pulse_pressure",
            "bp_category",
            "age_group",
            "bmi_category",
            "lifestyle_risk_score",
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
