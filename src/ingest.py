"""Dataset ingestion and harmonization module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.exception import CustomException
from src.logger import get_logger
from src.utils import ensure_directory, save_json


LOGGER = get_logger(__name__)


@dataclass
class DataQualityReport:
    """Structured data quality summary."""

    rows_before_cleaning: dict[str, int]
    rows_removed: dict[str, dict[str, int]]
    missing_value_rates: dict[str, dict[str, float]]
    outlier_counts: dict[str, dict[str, int]]


class DataIngestor:
    """Load, clean, and harmonize cardiovascular datasets."""

    def __init__(self, base_dir: Path | None = None) -> None:
        """Initialize ingestor paths.

        Args:
            base_dir: Optional project root path.

        Returns:
            None.
        """
        self.base_dir = base_dir or Path.cwd()
        self.raw_dir = self.base_dir / "data" / "raw"
        self.processed_dir = ensure_directory(self.base_dir / "data" / "processed")
        self.outputs_dir = ensure_directory(self.base_dir / "outputs")
        self.quality_report: DataQualityReport | None = None

    def load_framingham(self, path: Path) -> pd.DataFrame:
        """Load and map Framingham dataset to unified schema.

        Args:
            path: Input CSV path.

        Returns:
            Harmonized Framingham dataframe.
        """
        if not path.exists():
            raise FileNotFoundError(f"Framingham dataset not found at {path}")
        df = pd.read_csv(path).copy()
        df = df.rename(
            columns={
                "sex": "gender_bin",
                "age": "age_years",
                "sysBP": "systolic_bp",
                "diaBP": "diastolic_bp",
                "BMI": "bmi",
                "totChol": "cholesterol_raw",
                "currentSmoker": "smoke",
                "glucose": "glucose_raw",
                "TenYearCHD": "target",
            }
        )
        df["age_years"] = df["age_years"].astype(int)
        df = df.drop(
            columns=[
                "education",
                "cigsPerDay",
                "BPMeds",
                "prevalentStroke",
                "prevalentHyp",
                "diabetes",
                "heartRate",
            ]
        )
        df["alcohol"] = pd.NA
        df["active"] = pd.NA
        df["source"] = "framingham"
        return df

    def load_cardio(self, path: Path) -> pd.DataFrame:
        """Load and map cardio dataset to unified schema.

        Args:
            path: Input CSV path.

        Returns:
            Harmonized cardio dataframe.
        """
        if not path.exists():
            raise FileNotFoundError(f"Cardio dataset not found at {path}")
        df = pd.read_csv(path, sep=";").copy()
        df = df.drop(columns=["id"])
        df["age_years"] = (df["age"] / 365.25).astype(int)
        df["gender_bin"] = df["gender"].map({1: 0, 2: 1})
        df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
        df["cholesterol_raw"] = df["cholesterol"].map({1: 150, 2: 220, 3: 280})
        df["glucose_raw"] = df["gluc"].map({1: 80, 2: 120, 3: 180})
        df = df.rename(
            columns={
                "ap_hi": "systolic_bp",
                "ap_lo": "diastolic_bp",
                "smoke": "smoke",
                "alco": "alcohol",
                "active": "active",
                "cardio": "target",
            }
        )
        df["source"] = "cardio"
        return df[
            [
                "age_years",
                "gender_bin",
                "systolic_bp",
                "diastolic_bp",
                "bmi",
                "cholesterol_raw",
                "glucose_raw",
                "smoke",
                "alcohol",
                "active",
                "target",
                "source",
            ]
        ]

    def clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
        """Clean dataframe using physiological bounds and duplicates removal.

        Args:
            df: Input dataframe.

        Returns:
            Tuple of cleaned dataframe, rows removed by reason, and outlier counts.
        """
        frame = df.copy()
        conditions = {
            "systolic_bp": ~frame["systolic_bp"].between(60, 250),
            "diastolic_bp": ~frame["diastolic_bp"].between(40, 150),
            "bmi": ~frame["bmi"].between(10, 60),
            "age_years": ~frame["age_years"].between(18, 90),
        }
        rows_removed = {key: int(mask.sum()) for key, mask in conditions.items()}
        valid_mask = ~(conditions["systolic_bp"] | conditions["diastolic_bp"] | conditions["bmi"] | conditions["age_years"])
        cleaned = frame.loc[valid_mask].copy()
        duplicate_count = int(cleaned.duplicated().sum())
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        rows_removed["duplicates"] = duplicate_count
        outlier_counts = rows_removed.copy()
        for reason, count in rows_removed.items():
            LOGGER.info("Rows removed for %s: %s", reason, count)
        return cleaned, rows_removed, outlier_counts

    def harmonize(self) -> pd.DataFrame:
        """Load, clean, align, and save both datasets.

        Args:
            None.

        Returns:
            Concatenated harmonized dataframe.
        """
        framingham_path = self.raw_dir / "heart_disease.csv"
        cardio_path = self.raw_dir / "cardio_train.csv"
        try:
            framingham_raw = self.load_framingham(framingham_path)
            cardio_raw = self.load_cardio(cardio_path)
        except FileNotFoundError as exc:
            raise CustomException(str(exc)) from exc

        framingham_clean, framingham_removed, framingham_outliers = self.clean(framingham_raw)
        cardio_clean, cardio_removed, cardio_outliers = self.clean(cardio_raw)

        common_columns = [
            "age_years",
            "gender_bin",
            "systolic_bp",
            "diastolic_bp",
            "bmi",
            "cholesterol_raw",
            "glucose_raw",
            "smoke",
            "alcohol",
            "active",
            "target",
            "source",
        ]
        framingham_clean = framingham_clean[common_columns].copy()
        cardio_clean = cardio_clean[common_columns].copy()
        harmonized = pd.concat([framingham_clean, cardio_clean], ignore_index=True).reset_index(drop=True)

        output_path = self.processed_dir / "harmonized.csv"
        harmonized.to_csv(output_path, index=False)

        quality_report = DataQualityReport(
            rows_before_cleaning={
                "framingham": int(len(framingham_raw)),
                "cardio": int(len(cardio_raw)),
            },
            rows_removed={
                "framingham": framingham_removed,
                "cardio": cardio_removed,
            },
            missing_value_rates={
                "framingham": framingham_raw.isna().mean().round(4).to_dict(),
                "cardio": cardio_raw.isna().mean().round(4).to_dict(),
            },
            outlier_counts={
                "framingham": framingham_outliers,
                "cardio": cardio_outliers,
            },
        )
        self.quality_report = quality_report
        save_json(self.outputs_dir / "data_quality_report.json", quality_report)

        class_balance = harmonized["target"].value_counts(normalize=True).round(4).to_dict()
        LOGGER.info("Final harmonized shape: %s", harmonized.shape)
        LOGGER.info("Class balance: %s", class_balance)
        return harmonized


if __name__ == "__main__":
    ingestor = DataIngestor()
    try:
        dataframe = ingestor.harmonize()
        LOGGER.info("Ingest demo completed with shape %s", dataframe.shape)
    except CustomException as exc:
        LOGGER.error("Ingest demo failed: %s", exc)
