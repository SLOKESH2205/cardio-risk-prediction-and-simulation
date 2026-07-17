"""Run the full end-to-end cardiovascular risk platform pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluate import Evaluator
from src.explainability import SHAPExplainer
from src.features import FeatureEngineer
from src.ingest import DataIngestor
from src.logger import get_logger
from src.services.reporting import ReportingService
from src.train import ModelTrainer


LOGGER = get_logger(__name__)


def main() -> None:
    """Execute the project end-to-end pipeline."""
    parser = argparse.ArgumentParser(description="Run the cardiovascular risk intelligence pipeline")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run XGBoost RandomizedSearchCV tuning. This is now the default unless --quick is used.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip RandomizedSearchCV and use a fixed XGBoost configuration for faster local checks",
    )
    args = parser.parse_args()

    base_dir = Path.cwd()
    ingestor = DataIngestor(base_dir)
    harmonized_df = ingestor.harmonize()

    feature_engineer = FeatureEngineer()
    featured_df = feature_engineer.engineer(harmonized_df)

    reporting_service = ReportingService(base_dir)
    reporting_service.save_population_summary(featured_df)

    trainer = ModelTrainer(base_dir)
    training_results = trainer.train_all(featured_df, tune=not args.quick)

    if training_results["best_model_name"] in {"RandomForest", "XGBoost"}:
        try:
            explainer = SHAPExplainer(base_dir)
            feature_names = trainer.get_feature_names_from_pipeline(training_results["best_pipeline"])
            explainer.setup(training_results["best_pipeline"], training_results["X_train"])
            explainer.explain_global(
                training_results["X_test"],
                feature_names,
                base_dir / "outputs" / "shap_global.png",
            )
        except Exception as exc:  # pragma: no cover - explainability should not break training
            LOGGER.warning("SHAP report skipped: %s", exc)

    evaluator = Evaluator(base_dir)
    evaluator.full_report(
        training_results["best_pipeline"],
        training_results["X_test"],
        training_results["y_test"],
        training_results["best_model_name"],
        threshold=float(training_results["best_metrics"].get("threshold", 0.5)),
    )
    LOGGER.info("Pipeline complete. Run: streamlit run app.py")


if __name__ == "__main__":
    main()
