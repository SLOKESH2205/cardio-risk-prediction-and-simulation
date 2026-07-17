from __future__ import annotations

import pandas as pd
import numpy as np

from src.train import ModelTrainer, QuantileWinsorizer


def test_training_module_initializes_and_exposes_feature_columns() -> None:
    trainer = ModelTrainer(base_dir="d:/PROJECTS/VS CODE/heart_disease_prediction")
    assert trainer.NUMERICAL_FEATURES
    assert trainer.CATEGORICAL_FEATURES


def test_threshold_sweep_helper_ranges(tmp_path) -> None:
    trainer = ModelTrainer(base_dir=tmp_path)
    y_test = pd.Series([0, 1, 0, 1], name="target")
    y_prob = pd.Series([0.2, 0.3, 0.8, 0.9]).to_numpy()
    analysis = trainer._save_test_threshold_analysis(y_test, y_prob)
    assert analysis
    assert "best_f1_threshold" in analysis


def test_quantile_winsorizer_clips_extreme_values() -> None:
    transformer = QuantileWinsorizer(lower_quantile=0.25, upper_quantile=0.75)
    values = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 100.0]})

    transformed = transformer.fit_transform(values)

    assert transformed.min() >= 1.75
    assert transformed.max() <= 27.25


def test_improvement_report_helper_writes_markdown(tmp_path) -> None:
    trainer = ModelTrainer(base_dir=tmp_path)
    comparison_df = pd.DataFrame(
        [{"model": "XGBoost", "cv_auc": 0.82, "test_auc": 0.81, "f1": 0.72, "precision": 0.75, "recall": 0.68}]
    )
    feature_importance_df = pd.DataFrame([{"Feature": "num__age_years", "Importance": 0.3}])
    trainer.best_model_name = "XGBoost"
    trainer.best_metrics = {"model": "XGBoost", "cv_auc": 0.82, "test_auc": 0.81, "f1": 0.72, "precision": 0.75, "recall": 0.68}
    trainer.best_hyperparameters = {"n_estimators": 100, "cv_auc": 0.82}
    trainer.threshold_analysis = {"explanation": "Test explanation."}
    trainer._save_model_improvement_report(
        comparison_df,
        feature_importance_df,
        baseline_metrics={"model": "XGBoost", "cv_auc": 0.80, "test_auc": 0.80, "f1": 0.70, "precision": 0.74, "recall": 0.67},
    )
    report_path = tmp_path / "outputs" / "model_improvement_report.md"
    assert report_path.exists()
    contents = report_path.read_text(encoding="utf-8")
    assert "# Model Improvement Report" in contents
    assert "XGBoost" in contents


def test_error_analysis_saves_false_positive_and_false_negative_views(tmp_path) -> None:
    trainer = ModelTrainer(base_dir=tmp_path)
    X_test = pd.DataFrame(
        {
            "age_years": [40, 62, 55, 70],
            "systolic_bp": [118, 150, 125, 135],
            "diastolic_bp": [75, 95, 82, 88],
            "bmi": [22.0, 31.0, 26.0, 29.0],
        }
    )
    y_test = pd.Series([0, 0, 1, 1])
    y_prob = np.array([0.2, 0.9, 0.1, 0.8])

    trainer._save_error_analysis(X_test, y_test, y_prob, threshold=0.5)

    assert (tmp_path / "outputs" / "top_false_positives.csv").exists()
    assert (tmp_path / "outputs" / "top_false_negatives.csv").exists()
    assert (tmp_path / "outputs" / "error_feature_contrasts.csv").exists()
    summary = (tmp_path / "outputs" / "error_analysis_summary.json").read_text(encoding="utf-8")
    assert "false_positive_count" in summary
    assert "false_negative_count" in summary


def test_learning_curve_diagnostics_are_interpretable(tmp_path, monkeypatch) -> None:
    trainer = ModelTrainer(base_dir=tmp_path)

    def fake_learning_curve(*args, **kwargs):
        return (
            np.array([10, 20, 30]),
            np.array([[0.90, 0.91], [0.91, 0.90], [0.92, 0.91]]),
            np.array([[0.70, 0.71], [0.72, 0.71], [0.73, 0.72]]),
        )

    monkeypatch.setattr("src.train.learning_curve", fake_learning_curve)

    curve_df = trainer._save_learning_curve(
        pipeline=object(),
        X_train_validation=pd.DataFrame({"feature": [1, 2, 3]}),
        y_train_validation=pd.Series([0, 1, 0]),
    )

    assert len(curve_df) == 3
    diagnostics = (tmp_path / "outputs" / "learning_curve_diagnostics.json").read_text(encoding="utf-8")
    assert "overfitting" in diagnostics
