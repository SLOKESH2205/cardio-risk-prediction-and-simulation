"""Evaluation and model card module."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src.logger import get_logger
from src.utils import ensure_directory, save_json


LOGGER = get_logger(__name__)


@dataclass
class ModelCard:
    """Structured model card metadata."""

    model_name: str
    training_date: str
    dataset_size: int
    feature_count: int
    performance_metrics: dict[str, float]
    known_limitations: list[str]
    intended_use: str


class Evaluator:
    """Generate evaluation plots and model card outputs."""

    def __init__(self, base_dir: Path | None = None) -> None:
        """Initialize evaluator output paths.

        Args:
            base_dir: Optional project root path.

        Returns:
            None.
        """
        self.base_dir = base_dir or Path.cwd()
        self.outputs_dir = ensure_directory(self.base_dir / "outputs")

    def plot_roc(
        self,
        pipeline: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
        save_path: str | Path = "outputs/roc_curve.png",
    ) -> None:
        """Plot ROC curve.

        Args:
            pipeline: Fitted model pipeline.
            X_test: Test features.
            y_test: Test labels.
            model_name: Model display name.
            save_path: Output image path.

        Returns:
            None.
        """
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_value = roc_auc_score(y_test, y_prob)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} AUC = {auc_value:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(save_path), dpi=200)
        plt.close()

    def plot_confusion_matrix(
        self,
        pipeline: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        save_path: str | Path = "outputs/confusion_matrix.png",
    ) -> None:
        """Plot normalized confusion matrix.

        Args:
            pipeline: Fitted model pipeline.
            X_test: Test features.
            y_test: Test labels.
            save_path: Output image path.

        Returns:
            None.
        """
        y_pred = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, normalize="true")
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Normalized Confusion Matrix")
        plt.tight_layout()
        plt.savefig(Path(save_path), dpi=200)
        plt.close()

    def plot_pr_curve(
        self,
        pipeline: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        save_path: str | Path = "outputs/pr_curve.png",
    ) -> None:
        """Plot precision-recall curve.

        Args:
            pipeline: Fitted model pipeline.
            X_test: Test features.
            y_test: Test labels.
            save_path: Output image path.

        Returns:
            None.
        """
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        average_precision = average_precision_score(y_test, y_prob)
        plt.figure(figsize=(7, 5))
        plt.plot(recall, precision, linewidth=2, label=f"AP = {average_precision:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(save_path), dpi=200)
        plt.close()

    def full_report(
        self,
        pipeline: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
    ) -> dict[str, float]:
        """Generate full evaluation report and model card.

        Args:
            pipeline: Fitted model pipeline.
            X_test: Test features.
            y_test: Test labels.
            model_name: Model display name.

        Returns:
            Metrics dictionary.
        """
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        report_text = classification_report(y_test, y_pred)
        auc_value = float(roc_auc_score(y_test, y_prob))
        average_precision = float(average_precision_score(y_test, y_prob))
        LOGGER.info("Classification report for %s:\n%s", model_name, report_text)
        LOGGER.info("AUC %.4f | Average precision %.4f", auc_value, average_precision)

        self.plot_roc(pipeline, X_test, y_test, model_name, self.outputs_dir / "roc_curve.png")
        self.plot_confusion_matrix(pipeline, X_test, y_test, self.outputs_dir / "confusion_matrix.png")
        self.plot_pr_curve(pipeline, X_test, y_test, self.outputs_dir / "pr_curve.png")

        metrics = {
            "auc": auc_value,
            "average_precision": average_precision,
        }
        model_card = ModelCard(
            model_name=model_name,
            training_date=datetime.utcnow().isoformat(),
            dataset_size=int(len(X_test)),
            feature_count=int(X_test.shape[1]),
            performance_metrics=metrics,
            known_limitations=[
                "Model is trained on observational datasets and is not a diagnostic device.",
                "Framingham lacks alcohol and activity fields available in the cardio dataset.",
                "Cross-population transport may degrade for unseen clinical distributions.",
            ],
            intended_use="Educational risk stratification, analytics dashboards, and ML portfolio demonstration.",
        )
        save_json(self.outputs_dir / "model_card.json", model_card)
        return metrics


if __name__ == "__main__":
    LOGGER.info("Run run_pipeline.py to generate evaluation outputs.")
