"""Utilities for generating richer structured outputs for the app and reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import ensure_directory, save_json


class ReportingService:
    """Create structured insights used by the UI and report generation."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        self.outputs_dir = ensure_directory(self.base_dir / "outputs")

    def save_population_summary(self, df: pd.DataFrame) -> None:
        """Persist an aggregated population summary for the app."""
        summary = {
            "sample_size": int(len(df)),
            "positive_rate": round(float(df["target"].mean()), 4) if "target" in df.columns else None,
            "mean_age": round(float(df["age_years"].mean()), 2) if "age_years" in df.columns else None,
            "mean_bp": round(float(df["systolic_bp"].mean()), 2) if "systolic_bp" in df.columns else None,
        }
        save_json(self.outputs_dir / "population_summary.json", summary)

