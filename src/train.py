"""Model training module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from src.features import FeatureEngineer
from src.logger import get_logger
from src.utils import ensure_directory, save_joblib, save_json


LOGGER = get_logger(__name__)


class ModelTrainer:
    """Train cardiovascular risk models and compare performance."""

    NUMERICAL_FEATURES = [
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

    CATEGORICAL_FEATURES = ["bp_category", "age_group", "bmi_category", "gender_bin"]

    def __init__(self, base_dir: Path | None = None) -> None:
        """Initialize trainer paths and helpers.

        Args:
            base_dir: Optional project root path.

        Returns:
            None.
        """
        self.base_dir = base_dir or Path.cwd()
        self.models_dir = ensure_directory(self.base_dir / "models")
        self.outputs_dir = ensure_directory(self.base_dir / "outputs")
        self.feature_engineer = FeatureEngineer()
        self.best_model_name: str | None = None
        self.best_metrics: dict[str, Any] | None = None
        self.best_pipeline: Pipeline | None = None
        self.model_comparison: pd.DataFrame | None = None

    def build_pipeline(self, model: Any) -> Pipeline:
        """Create preprocessing and model pipeline.

        Args:
            model: Sklearn-compatible estimator.

        Returns:
            Full sklearn pipeline.
        """
        num_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        cat_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]
        )
        preprocessor = ColumnTransformer(
            [
                ("num", num_pipeline, self.NUMERICAL_FEATURES),
                ("cat", cat_pipeline, self.CATEGORICAL_FEATURES),
            ]
        )
        return Pipeline([("preprocessor", preprocessor), ("classifier", model)])

    def _evaluate_single_model(
        self,
        name: str,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> dict[str, Any]:
        """Train and evaluate one model.

        Args:
            name: Model name.
            pipeline: Pipeline instance.
            X_train: Training features.
            X_test: Test features.
            y_train: Training labels.
            y_test: Test labels.

        Returns:
            Metrics dictionary.
        """
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        mean_cv_auc = float(
            cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1).mean()
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "model": name,
            "cv_auc": mean_cv_auc,
            "test_auc": float(roc_auc_score(y_test, y_prob)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        LOGGER.info("Model %s metrics: %s", name, metrics)
        return metrics

    def train_all(self, df: pd.DataFrame) -> dict[str, Any]:
        """Train all requested models and save best pipeline.

        Args:
            df: Feature-engineered dataframe.

        Returns:
            Dictionary containing fitted pipelines and metrics.
        """
        feature_names = self.NUMERICAL_FEATURES + self.CATEGORICAL_FEATURES
        X = df[feature_names].copy()
        y = df["target"].copy()
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            stratify=y,
            test_size=0.2,
            random_state=42,
        )

        model_registry = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
            "XGBoost": XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
            ),
        }

        results: dict[str, Any] = {}
        comparison_rows: list[dict[str, Any]] = []

        for name, model in model_registry.items():
            pipeline = self.build_pipeline(model)
            metrics = self._evaluate_single_model(name, pipeline, X_train, X_test, y_train, y_test)
            results[name] = {
                "pipeline": pipeline,
                "metrics": metrics,
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            }
            comparison_rows.append(metrics)

        comparison_df = pd.DataFrame(comparison_rows).sort_values("test_auc", ascending=False).reset_index(drop=True)
        self.model_comparison = comparison_df
        LOGGER.info("Model comparison table:\n%s", comparison_df.to_string(index=False))
        comparison_df.to_csv(self.outputs_dir / "model_comparison.csv", index=False)

        best_row = comparison_df.iloc[0].to_dict()
        self.best_model_name = str(best_row["model"])
        self.best_metrics = best_row
        self.best_pipeline = results[self.best_model_name]["pipeline"]
        save_joblib(self.models_dir / "pipeline.pkl", self.best_pipeline)
        save_json(self.outputs_dir / "best_model_metrics.json", best_row)

        risk_reference = pd.DataFrame(
            {
                "risk_score": self.best_pipeline.predict_proba(X)[:, 1],
                "target": y.to_numpy(),
                "source": df["source"].to_numpy(),
            }
        )
        risk_reference.to_csv(self.outputs_dir / "risk_reference.csv", index=False)

        return {
            "pipelines": {name: details["pipeline"] for name, details in results.items()},
            "metrics": {name: details["metrics"] for name, details in results.items()},
            "best_model_name": self.best_model_name,
            "best_pipeline": self.best_pipeline,
            "best_metrics": self.best_metrics,
            "X_train": results[self.best_model_name]["X_train"],
            "X_test": results[self.best_model_name]["X_test"],
            "y_train": results[self.best_model_name]["y_train"],
            "y_test": results[self.best_model_name]["y_test"],
        }

    def cross_dataset_eval(
        self,
        pipeline: Pipeline,
        framingham_df: pd.DataFrame,
        cardio_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Evaluate generalization across the two source datasets.

        Args:
            pipeline: Reference pipeline whose classifier type is reused.
            framingham_df: Framingham dataframe.
            cardio_df: Cardio dataframe.

        Returns:
            Cross-dataset metrics dictionary.
        """
        framingham_features = self.feature_engineer.engineer(framingham_df)
        cardio_features = self.feature_engineer.engineer(cardio_df)
        classifier = pipeline.named_steps["classifier"]
        classifier_params = classifier.get_params()
        classifier_type = classifier.__class__

        cardio_to_frame_pipeline = self.build_pipeline(classifier_type(**classifier_params))
        cardio_to_frame_pipeline.fit(
            cardio_features[self.NUMERICAL_FEATURES + self.CATEGORICAL_FEATURES],
            cardio_features["target"],
        )
        cardio_to_frame_auc = float(
            roc_auc_score(
                framingham_features["target"],
                cardio_to_frame_pipeline.predict_proba(
                    framingham_features[self.NUMERICAL_FEATURES + self.CATEGORICAL_FEATURES]
                )[:, 1],
            )
        )

        frame_to_cardio_pipeline = self.build_pipeline(classifier_type(**classifier_params))
        frame_to_cardio_pipeline.fit(
            framingham_features[self.NUMERICAL_FEATURES + self.CATEGORICAL_FEATURES],
            framingham_features["target"],
        )
        frame_to_cardio_auc = float(
            roc_auc_score(
                cardio_features["target"],
                frame_to_cardio_pipeline.predict_proba(
                    cardio_features[self.NUMERICAL_FEATURES + self.CATEGORICAL_FEATURES]
                )[:, 1],
            )
        )

        base_auc = float(self.best_metrics["test_auc"]) if self.best_metrics else max(cardio_to_frame_auc, frame_to_cardio_auc)
        avg_cross_auc = (cardio_to_frame_auc + frame_to_cardio_auc) / 2
        auc_drop = max(base_auc - avg_cross_auc, 0.0)
        auc_drop_pct = auc_drop / base_auc * 100 if base_auc else 0.0
        if auc_drop_pct < 5:
            generalization = "good"
        elif auc_drop_pct < 12:
            generalization = "moderate"
        else:
            generalization = "poor"

        LOGGER.info("Train cardio -> test Framingham AUC: %.4f", cardio_to_frame_auc)
        LOGGER.info("Train Framingham -> test cardio AUC: %.4f", frame_to_cardio_auc)
        LOGGER.info(
            "AUC drop of %.2f%% indicates %s generalization",
            auc_drop_pct,
            generalization,
        )

        results = {
            "cardio_to_framingham_auc": cardio_to_frame_auc,
            "framingham_to_cardio_auc": frame_to_cardio_auc,
            "auc_drop_pct": auc_drop_pct,
            "generalization": generalization,
        }
        save_json(self.outputs_dir / "cross_dataset_eval.json", results)
        pd.DataFrame(
            [
                {"direction": "cardio_to_framingham", "auc": cardio_to_frame_auc},
                {"direction": "framingham_to_cardio", "auc": frame_to_cardio_auc},
            ]
        ).to_csv(self.outputs_dir / "cross_dataset_eval.csv", index=False)
        return results

    def get_feature_names_from_pipeline(self, pipeline: Pipeline) -> list[str]:
        """Extract transformed feature names from the preprocessor.

        Args:
            pipeline: Fitted sklearn pipeline.

        Returns:
            Ordered feature names after preprocessing.
        """
        preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
        return list(preprocessor.get_feature_names_out())


if __name__ == "__main__":
    harmonized_path = Path.cwd() / "data" / "processed" / "harmonized.csv"
    if harmonized_path.exists():
        df = pd.read_csv(harmonized_path)
        engineered = FeatureEngineer().engineer(df)
        trainer = ModelTrainer()
        trainer.train_all(engineered)
    else:
        LOGGER.info("Run ingest.py first to create harmonized.csv.")
