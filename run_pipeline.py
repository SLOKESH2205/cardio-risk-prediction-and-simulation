"""Run the full cardiovascular risk platform pipeline."""

from __future__ import annotations

from pathlib import Path

from src.evaluate import Evaluator
from src.explainability import SHAPExplainer
from src.features import FeatureEngineer
from src.ingest import DataIngestor
from src.logger import get_logger
from src.segmentation import PatientSegmenter
from src.train import ModelTrainer


LOGGER = get_logger(__name__)


def main() -> None:
    """Execute the project end-to-end pipeline.

    Args:
        None.

    Returns:
        None.
    """
    base_dir = Path.cwd()
    ingestor = DataIngestor(base_dir)
    harmonized_df = ingestor.harmonize()

    feature_engineer = FeatureEngineer()
    featured_df = feature_engineer.engineer(harmonized_df)

    segmenter = PatientSegmenter(base_dir)
    optimal_k = segmenter.find_optimal_k(featured_df)
    LOGGER.info("Optimal k suggested: %s", optimal_k)
    segmented_df = segmenter.fit(featured_df, k=3)
    cluster_profiles = segmenter.profile_clusters(segmented_df)
    LOGGER.info("Cluster profiles ready: %s", cluster_profiles)

    trainer = ModelTrainer(base_dir)
    training_results = trainer.train_all(segmented_df)
    framingham_df = segmented_df[segmented_df["source"] == "framingham"].copy()
    cardio_df = segmented_df[segmented_df["source"] == "cardio"].copy()
    trainer.cross_dataset_eval(training_results["best_pipeline"], framingham_df, cardio_df)

    explainer = SHAPExplainer(base_dir)
    feature_names = trainer.get_feature_names_from_pipeline(training_results["best_pipeline"])
    explainer.setup(training_results["best_pipeline"], training_results["X_train"])
    explainer.explain_global(
        training_results["X_test"],
        feature_names,
        base_dir / "outputs" / "shap_global.png",
    )

    evaluator = Evaluator(base_dir)
    evaluator.full_report(
        training_results["best_pipeline"],
        training_results["X_test"],
        training_results["y_test"],
        training_results["best_model_name"],
    )
    LOGGER.info("Pipeline complete. Run: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()

