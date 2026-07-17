"""Training module for the cardiovascular risk prediction project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from src.features import FeatureEngineer
from src.logger import get_logger
from src.utils import ensure_directory, save_joblib, save_json


LOGGER = get_logger(__name__)


BASELINE_METRICS = {
    "cv_auc": 0.8036,
    "test_auc": 0.8091,
    "precision": 0.7580,
    "recall": 0.6810,
    "f1": 0.7175,
}


class QuantileWinsorizer(BaseEstimator, TransformerMixin):
    """Clip numeric columns to train-fold quantiles to reduce outlier leverage."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: Any, y: Any = None) -> "QuantileWinsorizer":
        values = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(values, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.nanquantile(values, self.upper_quantile, axis=0)
        return self

    def transform(self, X: Any) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is None:
            return np.asarray([], dtype=object)
        return np.asarray(input_features, dtype=object)


class ModelTrainer:
    """Train Logistic Regression, Random Forest, and XGBoost candidates."""

    NUMERICAL_FEATURES = [
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
    CATEGORICAL_FEATURES = ["gender_bin", "bmi_category", "bp_category", "age_group"]

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        self.models_dir = ensure_directory(self.base_dir / "models")
        self.outputs_dir = ensure_directory(self.base_dir / "outputs")
        self.feature_engineer = FeatureEngineer()
        self.feature_columns: list[str] = []
        self.best_model_name: str | None = None
        self.best_pipeline: Pipeline | None = None
        self.best_metrics: dict[str, Any] | None = None
        self.best_hyperparameters: dict[str, Any] = {}
        self.best_threshold: float = 0.5
        self.threshold_analysis: dict[str, Any] = {}
        self.calibration_analysis: dict[str, Any] = {}

    def _select_feature_columns(self, df: pd.DataFrame) -> list[str]:
        available = [column for column in self.NUMERICAL_FEATURES + self.CATEGORICAL_FEATURES if column in df.columns]
        self.feature_columns = available
        return available

    def build_pipeline(
        self,
        model: Any,
        feature_columns: list[str] | None = None,
        preprocessing_strategy: str = "winsor_1_99",
    ) -> Pipeline:
        """Create a preprocessing and classifier pipeline."""
        selected_columns = feature_columns if feature_columns is not None else self.feature_columns
        numerical_features = [column for column in self.NUMERICAL_FEATURES if column in selected_columns]
        categorical_features = [column for column in self.CATEGORICAL_FEATURES if column in selected_columns]
        numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if preprocessing_strategy == "winsor_1_99":
            numeric_steps.append(("winsorizer", QuantileWinsorizer(lower_quantile=0.01, upper_quantile=0.99)))
        elif preprocessing_strategy == "clip_5_95":
            numeric_steps.append(("clipper", QuantileWinsorizer(lower_quantile=0.05, upper_quantile=0.95)))
        elif preprocessing_strategy != "none":
            raise ValueError(f"Unknown preprocessing strategy: {preprocessing_strategy}")
        numeric_steps.append(("scaler", StandardScaler()))
        preprocessor = ColumnTransformer(
            [
                (
                    "num",
                    Pipeline(numeric_steps),
                    numerical_features,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                        ]
                    ),
                    categorical_features,
                ),
            ]
        )
        return Pipeline([("preprocessor", preprocessor), ("classifier", model)])

    def _scale_pos_weight(self, y_train: pd.Series) -> float:
        """Calculate negative-to-positive class weight for XGBoost."""
        positive_count = int((y_train == 1).sum())
        negative_count = int((y_train == 0).sum())
        return negative_count / positive_count if positive_count else 1.0

    def _xgboost_search(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomizedSearchCV:
        """Tune XGBoost using training folds only."""
        base_scale_pos_weight = self._scale_pos_weight(y_train)
        scale_pos_weight_options = sorted(
            {
                round(base_scale_pos_weight * factor, 3)
                for factor in [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
            }
        )
        pipeline = self.build_pipeline(
            XGBClassifier(
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=42,
                scale_pos_weight=base_scale_pos_weight,
            )
        )
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions={
                "classifier__n_estimators": [100, 150, 200, 250, 300, 400],
                "classifier__max_depth": [2, 3, 4, 5, 6],
                "classifier__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
                "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
                "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "classifier__min_child_weight": [1, 2, 3, 5, 7],
                "classifier__gamma": [0, 0.05, 0.1, 0.2, 0.5],
                "classifier__reg_alpha": [0, 0.01, 0.05, 0.1, 0.5],
                "classifier__reg_lambda": [0.5, 1.0, 1.5, 2.0, 3.0],
                "classifier__scale_pos_weight": scale_pos_weight_options,
            },
            n_iter=45,
            scoring={"f1": "f1", "recall": "recall", "roc_auc": "roc_auc"},
            cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42),
            refit="f1",
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best_params = {key.replace("classifier__", ""): value for key, value in search.best_params_.items()}
        best_params["base_scale_pos_weight"] = base_scale_pos_weight
        best_index = int(search.best_index_)
        best_params["cv_f1"] = float(search.cv_results_["mean_test_f1"][best_index])
        best_params["cv_recall"] = float(search.cv_results_["mean_test_recall"][best_index])
        best_params["cv_auc"] = float(search.cv_results_["mean_test_roc_auc"][best_index])
        self.best_hyperparameters = best_params
        save_json(self.outputs_dir / "best_hyperparameters.json", best_params)
        LOGGER.info("Best XGBoost parameters: %s", best_params)
        return search

    def _fit_xgboost_default(self, X_train: pd.DataFrame, y_train: pd.Series, cv: StratifiedKFold) -> tuple[Pipeline, float]:
        """Fit a simple XGBoost model when a quick run is requested."""
        scale_pos_weight = self._scale_pos_weight(y_train)
        pipeline = self.build_pipeline(
            XGBClassifier(
                n_estimators=250,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                min_child_weight=3,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=42,
                scale_pos_weight=scale_pos_weight,
            )
        )
        cv_auc = float(cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1).mean())
        pipeline.fit(X_train, y_train)
        params = {
            "n_estimators": 250,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "min_child_weight": 3,
            "scale_pos_weight": scale_pos_weight,
            "cv_auc": cv_auc,
            "cv_f1": float(cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1).mean()),
            "cv_recall": float(cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="recall", n_jobs=-1).mean()),
            "search": "fixed_default",
        }
        self.best_hyperparameters = params
        save_json(self.outputs_dir / "best_hyperparameters.json", params)
        return pipeline, cv_auc

    def _evaluate(self, name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, cv_auc: float) -> dict[str, Any]:
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        return {
            "model": name,
            "cv_auc": float(cv_auc),
            "test_auc": float(roc_auc_score(y_test, y_prob)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

    def _threshold_metrics(self, y_test: pd.Series, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
        y_pred = (y_prob >= threshold).astype(int)
        return {
            "threshold": float(threshold),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

    def _best_threshold(self, y_true: pd.Series, y_prob: np.ndarray, optimize_for: str = "f1") -> dict[str, Any]:
        thresholds = np.linspace(0.05, 0.95, 91)
        scored = [self._threshold_metrics(y_true, y_prob, threshold) for threshold in thresholds]
        if optimize_for == "recall":
            return max(scored, key=lambda row: (row["recall"], row["f1"], row["precision"]))
        return max(scored, key=lambda row: (row["f1"], row["recall"]))

    def _save_threshold_analysis(
        self,
        y_validation: pd.Series,
        validation_prob: np.ndarray,
        y_test: pd.Series,
        test_prob: np.ndarray,
    ) -> dict[str, Any]:
        best = self._best_threshold(y_validation, validation_prob, optimize_for="f1")
        threshold = float(best["threshold"])
        validation_default = self._threshold_metrics(y_validation, validation_prob, 0.5)
        test_default = self._threshold_metrics(y_test, test_prob, 0.5)
        test_selected = self._threshold_metrics(y_test, test_prob, threshold)
        improved = test_selected["f1"] > test_default["f1"]
        explanation = (
            "Threshold tuning improved test F1 after the cutoff was selected on the validation split, "
            "so the model found more true heart-risk cases without using test labels to choose the threshold."
            if improved
            else "The validation-selected threshold did not improve test F1, which means the default 0.50 cutoff "
            "was at least as strong on this final split."
        )
        payload = {
            "validation_threshold_0_50": validation_default,
            "validation_best_f1_threshold": best,
            "test_threshold_0_50": test_default,
            "test_at_validation_threshold": test_selected,
            "explanation": explanation,
        }
        self.best_threshold = threshold
        self.threshold_analysis = payload
        save_json(self.outputs_dir / "threshold_analysis.json", payload)
        return payload

    def _save_test_threshold_analysis(self, y_test: pd.Series, y_prob: np.ndarray) -> dict[str, Any]:
        """Compatibility helper for small unit tests."""
        thresholds = np.linspace(0.05, 0.95, 91)
        scored = [self._threshold_metrics(y_test, y_prob, threshold) for threshold in thresholds]
        best = max(scored, key=lambda row: row["f1"])
        default = self._threshold_metrics(y_test, y_prob, 0.5)
        improved = best["f1"] > default["f1"]
        explanation = (
            "Threshold tuning improved F1 because lowering or raising the cutoff changed the precision-recall balance "
            "so the model found more true heart-risk cases without adding too many false alarms."
            if improved
            else "Threshold tuning did not improve F1 because the default 0.50 cutoff already gave the best balance "
            "between precision and recall for this test split."
        )
        payload = {
            "threshold_0_50": default,
            "best_f1_threshold": best,
            "explanation": explanation,
        }
        self.best_threshold = float(best["threshold"])
        self.threshold_analysis = payload
        save_json(self.outputs_dir / "threshold_analysis.json", payload)
        return payload

    def _save_calibration_analysis(
        self,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
    ) -> dict[str, Any]:
        """Fit probability calibration diagnostics without replacing the explainable base pipeline."""
        calibrated = CalibratedClassifierCV(estimator=pipeline, method="sigmoid", cv=3)
        calibrated.fit(X_train, y_train)
        base_prob = pipeline.predict_proba(X_validation)[:, 1]
        calibrated_prob = calibrated.predict_proba(X_validation)[:, 1]
        base_threshold = self._best_threshold(y_validation, base_prob)
        calibrated_threshold = self._best_threshold(y_validation, calibrated_prob)
        analysis = {
            "method": "sigmoid",
            "selected_for_deployment": False,
            "reason": "Calibration is reported as a reliability diagnostic; the saved pipeline stays directly explainable for SHAP and feature importance.",
            "base_validation_auc": float(roc_auc_score(y_validation, base_prob)),
            "calibrated_validation_auc": float(roc_auc_score(y_validation, calibrated_prob)),
            "base_best_f1": float(base_threshold["f1"]),
            "calibrated_best_f1": float(calibrated_threshold["f1"]),
            "base_best_threshold": float(base_threshold["threshold"]),
            "calibrated_best_threshold": float(calibrated_threshold["threshold"]),
        }
        self.calibration_analysis = analysis
        save_json(self.outputs_dir / "calibration_analysis.json", analysis)
        return analysis

    def _validation_summary(self, pipeline: Pipeline, X_validation: pd.DataFrame, y_validation: pd.Series) -> dict[str, Any]:
        """Return validation AUC and best-threshold classification metrics."""
        probabilities = pipeline.predict_proba(X_validation)[:, 1]
        best = self._best_threshold(y_validation, probabilities)
        return {
            "validation_auc": float(roc_auc_score(y_validation, probabilities)),
            "best_validation_threshold": float(best["threshold"]),
            "best_validation_precision": float(best["precision"]),
            "best_validation_recall": float(best["recall"]),
            "best_validation_f1": float(best["f1"]),
        }

    def _classifier_from_pipeline(self, pipeline: Pipeline) -> Any:
        """Clone the selected classifier while leaving preprocessing experiments free to vary."""
        return clone(pipeline.named_steps["classifier"])

    def _save_preprocessing_ablation(
        self,
        pipeline: Pipeline,
        feature_columns: list[str],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
    ) -> tuple[pd.DataFrame, str]:
        """Compare simple preprocessing options and let validation F1 choose."""
        rows: list[dict[str, Any]] = []
        for strategy in ["none", "winsor_1_99", "clip_5_95"]:
            candidate = self.build_pipeline(
                self._classifier_from_pipeline(pipeline),
                feature_columns=feature_columns,
                preprocessing_strategy=strategy,
            )
            candidate.fit(X_train, y_train)
            row = {"preprocessing_strategy": strategy}
            row.update(self._validation_summary(candidate, X_validation, y_validation))
            rows.append(row)
        ablation_df = pd.DataFrame(rows).sort_values(
            ["best_validation_f1", "validation_auc"],
            ascending=False,
        ).reset_index(drop=True)
        ablation_df.to_csv(self.outputs_dir / "preprocessing_ablation.csv", index=False)
        best_row = ablation_df.iloc[0]
        save_json(
            self.outputs_dir / "preprocessing_ablation_summary.json",
            {
                "selected_strategy": str(best_row["preprocessing_strategy"]),
                "selected_validation_f1": float(best_row["best_validation_f1"]),
                "selected_validation_auc": float(best_row["validation_auc"]),
                "rule": "Choose the preprocessing strategy with the best validation F1, using validation AUC as the tie-breaker.",
                "strategies_compared": ablation_df["preprocessing_strategy"].tolist(),
            },
        )
        return ablation_df, str(ablation_df.iloc[0]["preprocessing_strategy"])

    def _save_feature_ablation(
        self,
        pipeline: Pipeline,
        feature_columns: list[str],
        preprocessing_strategy: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Retrain one-feature-drop variants and recommend drops only when validation improves."""
        rows: list[dict[str, Any]] = []
        baseline = self.build_pipeline(
            self._classifier_from_pipeline(pipeline),
            feature_columns=feature_columns,
            preprocessing_strategy=preprocessing_strategy,
        )
        baseline.fit(X_train, y_train)
        baseline_summary = self._validation_summary(baseline, X_validation, y_validation)
        rows.append({"variant": "all_features", "dropped_feature": "", "decision": "baseline", **baseline_summary})

        engineered_features = [
            feature for feature in self.feature_engineer.get_feature_names()["engineered"] if feature in feature_columns
        ]
        recommended_drops: list[str] = []
        for feature in engineered_features:
            candidate_columns = [column for column in feature_columns if column != feature]
            candidate = self.build_pipeline(
                self._classifier_from_pipeline(pipeline),
                feature_columns=candidate_columns,
                preprocessing_strategy=preprocessing_strategy,
            )
            candidate.fit(X_train[candidate_columns], y_train)
            summary = self._validation_summary(candidate, X_validation[candidate_columns], y_validation)
            f1_delta = summary["best_validation_f1"] - baseline_summary["best_validation_f1"]
            auc_delta = summary["validation_auc"] - baseline_summary["validation_auc"]
            should_drop = f1_delta > 0.0005 or (abs(f1_delta) <= 0.0005 and auc_delta > 0.0005)
            if should_drop:
                recommended_drops.append(feature)
            rows.append(
                {
                    "variant": f"drop_{feature}",
                    "dropped_feature": feature,
                    "decision": "drop" if should_drop else "keep",
                    "f1_delta_vs_all": float(f1_delta),
                    "auc_delta_vs_all": float(auc_delta),
                    **summary,
                }
            )
        ablation_df = pd.DataFrame(rows).sort_values("best_validation_f1", ascending=False).reset_index(drop=True)
        ablation_df.to_csv(self.outputs_dir / "feature_ablation.csv", index=False)
        retained_columns = [column for column in feature_columns if column not in recommended_drops]
        save_json(
            self.outputs_dir / "selected_features.json",
            {
                "recommended_drops": recommended_drops,
                "retained_features": retained_columns,
                "baseline_validation_f1": float(baseline_summary["best_validation_f1"]),
                "baseline_validation_auc": float(baseline_summary["validation_auc"]),
                "best_variant": str(ablation_df.iloc[0]["variant"]),
                "best_variant_validation_f1": float(ablation_df.iloc[0]["best_validation_f1"]),
                "best_variant_validation_auc": float(ablation_df.iloc[0]["validation_auc"]),
                "rule": "Drop engineered features only when one-feature retraining improves validation F1, or ties F1 and improves validation AUC.",
            },
        )
        return ablation_df, retained_columns

    def _save_learning_curve(
        self,
        pipeline: Pipeline,
        X_train_validation: pd.DataFrame,
        y_train_validation: pd.Series,
    ) -> pd.DataFrame:
        """Save train-vs-validation scores to diagnose overfitting or data bottlenecks."""
        train_sizes, train_scores, validation_scores = learning_curve(
            pipeline,
            X_train_validation,
            y_train_validation,
            train_sizes=np.linspace(0.2, 1.0, 5),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="f1",
            n_jobs=-1,
        )
        rows = []
        for index, size in enumerate(train_sizes):
            rows.append(
                {
                    "train_size": int(size),
                    "train_f1_mean": float(np.mean(train_scores[index])),
                    "train_f1_std": float(np.std(train_scores[index])),
                    "validation_f1_mean": float(np.mean(validation_scores[index])),
                    "validation_f1_std": float(np.std(validation_scores[index])),
                    "generalization_gap": float(np.mean(train_scores[index]) - np.mean(validation_scores[index])),
                }
            )
        curve_df = pd.DataFrame(rows)
        curve_df.to_csv(self.outputs_dir / "learning_curve.csv", index=False)
        final_row = curve_df.iloc[-1]
        first_row = curve_df.iloc[0]
        validation_gain = float(final_row["validation_f1_mean"] - first_row["validation_f1_mean"])
        final_gap = float(final_row["generalization_gap"])
        if final_gap >= 0.08:
            diagnosis = "overfitting"
            explanation = "Training F1 is materially higher than validation F1, so regularization or simpler features may help."
        elif validation_gain < 0.01:
            diagnosis = "data_or_feature_bottleneck"
            explanation = "Validation F1 barely improves as more data is added, which suggests the current feature signal is the main bottleneck."
        else:
            diagnosis = "reasonable_fit"
            explanation = "Training and validation F1 move together without a large final gap."
        save_json(
            self.outputs_dir / "learning_curve_diagnostics.json",
            {
                "diagnosis": diagnosis,
                "explanation": explanation,
                "final_train_f1_mean": float(final_row["train_f1_mean"]),
                "final_validation_f1_mean": float(final_row["validation_f1_mean"]),
                "final_generalization_gap": final_gap,
                "validation_f1_gain_from_smallest_to_full_train": validation_gain,
            },
        )
        return curve_df

    def _save_error_analysis(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_prob: np.ndarray,
        threshold: float,
    ) -> None:
        """Persist false-positive/false-negative summaries for inspection."""
        frame = X_test.copy()
        frame["actual"] = y_test.to_numpy()
        frame["probability"] = y_prob
        frame["predicted"] = (y_prob >= threshold).astype(int)
        frame["error_type"] = "correct"
        frame.loc[(frame["actual"] == 0) & (frame["predicted"] == 1), "error_type"] = "false_positive"
        frame.loc[(frame["actual"] == 1) & (frame["predicted"] == 0), "error_type"] = "false_negative"

        false_positives = frame.loc[frame["error_type"] == "false_positive"].copy()
        false_negatives = frame.loc[frame["error_type"] == "false_negative"].copy()
        false_positives.sort_values("probability", ascending=False).head(100).to_csv(
            self.outputs_dir / "top_false_positives.csv",
            index=False,
        )
        false_negatives.sort_values("probability", ascending=True).head(100).to_csv(
            self.outputs_dir / "top_false_negatives.csv",
            index=False,
        )
        pd.concat(
            [
                false_positives.sort_values("probability", ascending=False).head(50),
                false_negatives.sort_values("probability", ascending=True).head(50),
            ],
            axis=0,
        ).to_csv(self.outputs_dir / "top_prediction_errors.csv", index=False)
        summary_rows = []
        numeric_columns = [column for column in self.NUMERICAL_FEATURES if column in frame.columns]
        for error_type in ["false_positive", "false_negative", "correct"]:
            subset = frame.loc[frame["error_type"] == error_type]
            row: dict[str, Any] = {"error_type": error_type, "count": int(len(subset))}
            for column in numeric_columns:
                row[f"{column}_mean"] = float(subset[column].mean()) if len(subset) else 0.0
            summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(self.outputs_dir / "error_analysis_summary.csv", index=False)
        correct_row = summary_df.loc[summary_df["error_type"] == "correct"].iloc[0]
        contrast_rows = []
        for error_type in ["false_positive", "false_negative"]:
            error_row = summary_df.loc[summary_df["error_type"] == error_type].iloc[0]
            for column in numeric_columns:
                contrast_rows.append(
                    {
                        "error_type": error_type,
                        "feature": column,
                        "error_mean": float(error_row[f"{column}_mean"]),
                        "correct_mean": float(correct_row[f"{column}_mean"]),
                        "mean_delta_vs_correct": float(error_row[f"{column}_mean"] - correct_row[f"{column}_mean"]),
                    }
                )
        pd.DataFrame(contrast_rows).sort_values(
            "mean_delta_vs_correct",
            key=lambda values: values.abs(),
            ascending=False,
        ).to_csv(self.outputs_dir / "error_feature_contrasts.csv", index=False)
        save_json(
            self.outputs_dir / "error_analysis_summary.json",
            {
                "threshold": float(threshold),
                "false_positive_count": int(len(false_positives)),
                "false_negative_count": int(len(false_negatives)),
                "files": [
                    "outputs/top_false_positives.csv",
                    "outputs/top_false_negatives.csv",
                    "outputs/error_feature_contrasts.csv",
                ],
            },
        )

    def _save_feature_importance(self, pipeline: Pipeline) -> pd.DataFrame:
        feature_names = self.get_feature_names_from_pipeline(pipeline)
        classifier = pipeline.named_steps["classifier"]
        if hasattr(classifier, "feature_importances_"):
            importance = np.asarray(classifier.feature_importances_, dtype=float)
        elif hasattr(classifier, "coef_"):
            importance = np.abs(np.asarray(classifier.coef_[0], dtype=float))
        else:
            importance = np.zeros(len(feature_names), dtype=float)
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
        importance_df = importance_df.sort_values("Importance", ascending=False).reset_index(drop=True)
        importance_df.to_csv(self.outputs_dir / "feature_importance.csv", index=False)
        return importance_df

    def _load_baseline_metrics(self) -> dict[str, Any]:
        baseline_path = self.outputs_dir / "baseline_best_model_metrics.json"
        if baseline_path.exists():
            with baseline_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                return payload
        save_json(baseline_path, BASELINE_METRICS)
        return BASELINE_METRICS.copy()

    def _metric_delta(self, baseline: float, current: float) -> str:
        delta = current - baseline
        if abs(delta) < 0.00005:
            return "No Change."
        return f"{delta:+.4f}"

    def _save_model_improvement_report(
        self,
        comparison_df: pd.DataFrame,
        feature_importance_df: pd.DataFrame,
        baseline_metrics: dict[str, Any],
    ) -> None:
        top_features = feature_importance_df.head(10)
        best_params = {key: value for key, value in self.best_hyperparameters.items() if key != "cv_auc"}
        lines = [
            "# Model Improvement Report",
            "",
            "## Files modified",
            "- src/ingest.py",
            "- src/features.py",
            "- src/train.py",
            "- run_pipeline.py",
            "- requirements.txt",
            "- tests/test_features.py",
            "- tests/test_prediction.py",
            "- tests/test_training.py",
            "",
            "## Why each modification was made",
            "- Data cleaning now removes rows where diastolic blood pressure is greater than or equal to systolic blood pressure.",
            "- Feature engineering now keeps only the approved simple clinical features.",
            "- Training now compares only Logistic Regression, Random Forest, and XGBoost.",
            "- XGBoost tuning uses RandomizedSearchCV on training folds only.",
            "- Numeric preprocessing imputes medians and clips train-fold outliers with 1st/99th percentile winsorization.",
            "- XGBoost uses scale_pos_weight to account for class imbalance without synthetic samples.",
            "- Model selection now uses validation F1 instead of AUC alone.",
            "- Threshold optimization now chooses the cutoff on validation and reports final test performance separately.",
            "- Calibration diagnostics compare the base model with sigmoid-calibrated probabilities.",
            "- Feature ablation retrains one-feature-drop variants and records retained features.",
            "- Preprocessing ablation compares no clipping, 1st/99th percentile winsorization, and 5th/95th percentile clipping.",
            "- Learning-curve and error-analysis files are saved for interview diagnostics.",
            "- Threshold analysis, feature importance, hyperparameters, and this report are saved as assignment outputs.",
            "",
            "## Metric comparison",
            "",
            "metric | baseline | new | delta",
            "--- | ---: | ---: | ---:",
        ]
        for metric in ["cv_auc", "test_auc", "precision", "recall", "f1"]:
            current = float((self.best_metrics or {}).get(metric, 0.0))
            baseline = float(baseline_metrics.get(metric, BASELINE_METRICS[metric]))
            lines.append(f"{metric} | {baseline:.4f} | {current:.4f} | {self._metric_delta(baseline, current)}")

        lines.extend(
            [
                "",
                "## Whether each modification improved metrics",
                f"- Best current model: {self.best_model_name}",
                f"- Validation AUC changed by {self._metric_delta(float(baseline_metrics.get('cv_auc', BASELINE_METRICS['cv_auc'])), float((self.best_metrics or {}).get('cv_auc', 0.0)))}.",
                f"- Test F1 changed by {self._metric_delta(float(baseline_metrics.get('f1', BASELINE_METRICS['f1'])), float((self.best_metrics or {}).get('f1', 0.0)))}.",
                "",
                "## Best XGBoost parameters",
                json.dumps(best_params, indent=2),
                "",
                "## Best threshold",
                f"- {self.best_threshold:.2f}",
                "",
                "## Interview explanation of threshold tuning",
                f"- {self.threshold_analysis.get('explanation', '')}",
                "",
                "## Calibration diagnostics",
                json.dumps(self.calibration_analysis, indent=2),
                "",
                "## Feature ablation",
                "- See outputs/feature_ablation.csv and outputs/selected_features.json.",
                "",
                "## Preprocessing ablation",
                "- See outputs/preprocessing_ablation.csv.",
                "",
                "## Learning curve and errors",
                "- See outputs/learning_curve.csv and outputs/learning_curve_diagnostics.json.",
                "- See outputs/error_analysis_summary.csv, outputs/error_feature_contrasts.csv, outputs/top_false_positives.csv, and outputs/top_false_negatives.csv.",
                "",
                "## Top 10 important features",
                "",
                "| Feature | Importance |",
                "| --- | ---: |",
            ]
        )
        for _, row in top_features.iterrows():
            lines.append(f"| {row['Feature']} | {float(row['Importance']):.6f} |")

        lines.extend(
            [
                "",
                "## Candidate models",
                "",
                "| Model | CV AUC | Test AUC | Precision | Recall | F1 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in comparison_df.iterrows():
            lines.append(
                f"| {row['model']} | {row['cv_auc']:.4f} | {row['test_auc']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |"
            )

        lines.extend(
            [
                "",
                "## Remaining limitations",
                "- This is an educational risk prediction model, not a medical diagnostic tool.",
                "- The two datasets use different feature definitions, so external validation would still be needed before real clinical use.",
                "- Threshold tuning was evaluated after the model was fixed; it should be rechecked for any new dataset.",
            ]
        )
        (self.outputs_dir / "model_improvement_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def train_all(self, df: pd.DataFrame, tune: bool = True) -> dict[str, Any]:
        """Train the approved model set and save project outputs."""
        engineered = self.feature_engineer.engineer(df.copy())
        feature_columns = self._select_feature_columns(engineered)
        X = engineered[feature_columns].copy()
        X = X.apply(pd.to_numeric, errors="coerce")
        y = engineered["target"].astype(int).copy()
        X_train_validation, X_test, y_train_validation, y_test = train_test_split(
            X,
            y,
            stratify=y,
            test_size=0.2,
            random_state=42,
        )
        X_train, X_validation, y_train, y_validation = train_test_split(
            X_train_validation,
            y_train_validation,
            stratify=y_train_validation,
            test_size=0.2,
            random_state=42,
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        candidates: dict[str, Pipeline] = {
            "LogisticRegression": self.build_pipeline(
                LogisticRegression(max_iter=4000, solver="lbfgs", class_weight="balanced", random_state=42)
            ),
            "RandomForest": self.build_pipeline(
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=None,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                )
            ),
        }
        comparison_rows: list[dict[str, Any]] = []
        results: dict[str, dict[str, Any]] = {}
        for name, pipeline in candidates.items():
            cv_auc = float(cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1).mean())
            pipeline.fit(X_train, y_train)
            validation_prob = pipeline.predict_proba(X_validation)[:, 1]
            validation_threshold = self._best_threshold(y_validation, validation_prob)
            metrics = self._evaluate(name, pipeline, X_validation, y_validation, cv_auc)
            metrics.update(
                {
                    "validation_threshold": float(validation_threshold["threshold"]),
                    "validation_threshold_f1": float(validation_threshold["f1"]),
                    "selection_score": float(validation_threshold["f1"]),
                }
            )
            comparison_rows.append(metrics)
            results[name] = {"pipeline": pipeline, "metrics": metrics}

        if tune:
            xgb_search = self._xgboost_search(X_train, y_train)
            xgb_pipeline = xgb_search.best_estimator_
            best_index = int(xgb_search.best_index_)
            xgb_cv_auc = float(xgb_search.cv_results_["mean_test_roc_auc"][best_index])
        else:
            xgb_pipeline, xgb_cv_auc = self._fit_xgboost_default(X_train, y_train, cv)
        xgb_validation_prob = xgb_pipeline.predict_proba(X_validation)[:, 1]
        xgb_validation_threshold = self._best_threshold(y_validation, xgb_validation_prob)
        xgb_metrics = self._evaluate("XGBoost", xgb_pipeline, X_validation, y_validation, xgb_cv_auc)
        xgb_metrics.update(
            {
                "validation_threshold": float(xgb_validation_threshold["threshold"]),
                "validation_threshold_f1": float(xgb_validation_threshold["f1"]),
                "selection_score": float(xgb_validation_threshold["f1"]),
            }
        )
        comparison_rows.append(xgb_metrics)
        results["XGBoost"] = {"pipeline": xgb_pipeline, "metrics": xgb_metrics}

        comparison_df = pd.DataFrame(comparison_rows).sort_values("selection_score", ascending=False).reset_index(drop=True)
        comparison_df.to_csv(self.outputs_dir / "model_comparison.csv", index=False)

        self.best_model_name = str(comparison_df.iloc[0]["model"])
        pre_refit_pipeline = results[self.best_model_name]["pipeline"]
        preprocessing_ablation_df, best_preprocessing_strategy = self._save_preprocessing_ablation(
            pre_refit_pipeline,
            feature_columns,
            X_train,
            y_train,
            X_validation,
            y_validation,
        )
        _ = preprocessing_ablation_df
        feature_ablation_df, retained_feature_columns = self._save_feature_ablation(
            pre_refit_pipeline,
            feature_columns,
            best_preprocessing_strategy,
            X_train,
            y_train,
            X_validation,
            y_validation,
        )
        _ = feature_ablation_df

        validation_pipeline = self.build_pipeline(
            self._classifier_from_pipeline(pre_refit_pipeline),
            feature_columns=retained_feature_columns,
            preprocessing_strategy=best_preprocessing_strategy,
        )
        validation_pipeline.fit(X_train[retained_feature_columns], y_train)
        selected_pipeline = self.build_pipeline(
            self._classifier_from_pipeline(pre_refit_pipeline),
            feature_columns=retained_feature_columns,
            preprocessing_strategy=best_preprocessing_strategy,
        )
        selected_pipeline.fit(X_train_validation[retained_feature_columns], y_train_validation)
        self.best_pipeline = selected_pipeline
        save_joblib(self.models_dir / "pipeline.pkl", self.best_pipeline)

        validation_prob = validation_pipeline.predict_proba(X_validation[retained_feature_columns])[:, 1]
        test_prob = self.best_pipeline.predict_proba(X_test[retained_feature_columns])[:, 1]
        threshold_analysis = self._save_threshold_analysis(y_validation, validation_prob, y_test, test_prob)
        threshold_metrics = threshold_analysis["test_at_validation_threshold"]
        default_test_metrics = self._threshold_metrics(y_test, test_prob, 0.5)
        validation_selection_metrics = self._evaluate(
            self.best_model_name,
            validation_pipeline,
            X_validation[retained_feature_columns],
            y_validation,
            float(results[self.best_model_name]["metrics"]["cv_auc"]),
        )
        validation_best = self._best_threshold(y_validation, validation_prob)
        validation_selection_metrics.update(
            {
                "validation_threshold": float(validation_best["threshold"]),
                "validation_threshold_f1": float(validation_best["f1"]),
                "selection_score": float(validation_best["f1"]),
                "preprocessing_strategy": best_preprocessing_strategy,
                "feature_count": len(retained_feature_columns),
            }
        )
        self.best_metrics = {
            "model": self.best_model_name,
            "cv_auc": float(results[self.best_model_name]["metrics"]["cv_auc"]),
            "test_auc": float(roc_auc_score(y_test, test_prob)),
            "precision": float(threshold_metrics["precision"]),
            "recall": float(threshold_metrics["recall"]),
            "f1": float(threshold_metrics["f1"]),
            "confusion_matrix": threshold_metrics["confusion_matrix"],
            "threshold": float(threshold_metrics["threshold"]),
            "default_threshold_metrics": {
                "model": self.best_model_name,
                "cv_auc": float(results[self.best_model_name]["metrics"]["cv_auc"]),
                "test_auc": float(roc_auc_score(y_test, test_prob)),
                "precision": float(default_test_metrics["precision"]),
                "recall": float(default_test_metrics["recall"]),
                "f1": float(default_test_metrics["f1"]),
                "confusion_matrix": default_test_metrics["confusion_matrix"],
            },
            "validation_selection_metrics": validation_selection_metrics,
            "preprocessing_strategy": best_preprocessing_strategy,
            "feature_count": len(retained_feature_columns),
        }
        self.best_metrics.update(
            {
                "calibration": self._save_calibration_analysis(
                    validation_pipeline,
                    X_train,
                    y_train,
                    X_validation[retained_feature_columns],
                    y_validation,
                )
            }
        )
        save_json(self.outputs_dir / "best_model_metrics.json", self.best_metrics)
        feature_importance_df = self._save_feature_importance(self.best_pipeline)
        self._save_learning_curve(
            self.best_pipeline,
            X_train_validation[retained_feature_columns],
            y_train_validation,
        )
        self._save_error_analysis(X_test, y_test, test_prob, float(threshold_metrics["threshold"]))

        risk_reference = pd.DataFrame(
            {
                "risk_score": self.best_pipeline.predict_proba(X)[:, 1],
                "target": y.to_numpy(),
                "source": engineered["source"].to_numpy(),
            }
        )
        risk_reference.to_csv(self.outputs_dir / "risk_reference.csv", index=False)

        baseline_metrics = self._load_baseline_metrics()
        self._save_model_improvement_report(comparison_df, feature_importance_df, baseline_metrics)

        LOGGER.info("Model comparison table:\n%s", comparison_df.to_string(index=False))
        LOGGER.info("Best model metrics: %s", self.best_metrics)
        return {
            "pipelines": {name: details["pipeline"] for name, details in results.items()},
            "metrics": {name: details["metrics"] for name, details in results.items()},
            "best_model_name": self.best_model_name,
            "best_pipeline": self.best_pipeline,
            "best_metrics": self.best_metrics,
            "X_train": X_train_validation,
            "X_test": X_test,
            "y_train": y_train_validation,
            "y_test": y_test,
        }

    def get_feature_names_from_pipeline(self, pipeline: Pipeline) -> list[str]:
        """Extract transformed feature names from the preprocessor."""
        preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
        return list(preprocessor.get_feature_names_out())


if __name__ == "__main__":
    harmonized_path = Path.cwd() / "data" / "processed" / "harmonized.csv"
    if harmonized_path.exists():
        dataframe = pd.read_csv(harmonized_path)
        trainer = ModelTrainer()
        trainer.train_all(dataframe)
    else:
        LOGGER.info("Run ingest.py first to create harmonized.csv.")
