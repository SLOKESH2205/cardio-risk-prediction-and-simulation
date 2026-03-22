"""SHAP explainability module and patient-facing explanation helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.logger import get_logger
from src.utils import ensure_directory


LOGGER = get_logger(__name__)


class SHAPExplainer:
    """Generate global and local SHAP-based explanations."""

    FEATURE_DESCRIPTIONS = {
        "age_years": ("age", "years"),
        "systolic_bp": ("systolic blood pressure", "mmHg"),
        "diastolic_bp": ("diastolic blood pressure", "mmHg"),
        "bmi": ("BMI", ""),
        "pulse_pressure": ("pulse pressure", "mmHg"),
        "cholesterol_raw": ("cholesterol level", "mg/dL"),
        "glucose_raw": ("glucose level", "mg/dL"),
        "lifestyle_risk_score": ("lifestyle risk score", "/4"),
        "bp_bmi_interaction": ("BP-BMI interaction", ""),
        "age_bp_interaction": ("age-BP interaction", ""),
        "bp_category": ("blood pressure category", ""),
        "age_group": ("age group", ""),
        "bmi_category": ("BMI category", ""),
        "gender_bin": ("gender", ""),
    }

    def __init__(self, base_dir: Path | None = None, random_state: int = 42) -> None:
        self.base_dir = base_dir or Path.cwd()
        self.outputs_dir = ensure_directory(self.base_dir / "outputs")
        self.random_state = random_state
        self.explainer: shap.TreeExplainer | None = None
        self.preprocessor: Any | None = None
        self.pipeline: Pipeline | None = None
        self.feature_names: list[str] = []
        self.numeric_feature_stds: pd.Series | None = None
        self.expected_value: float | None = None

    def setup(self, pipeline: Pipeline, X_train: pd.DataFrame) -> None:
        self.pipeline = pipeline
        model = pipeline.named_steps["classifier"]
        self.preprocessor = pipeline.named_steps["preprocessor"]
        _ = self.preprocessor.transform(X_train)
        self.explainer = shap.TreeExplainer(model)
        self.feature_names = list(self.preprocessor.get_feature_names_out())
        self.numeric_feature_stds = X_train.std(numeric_only=True)
        expected_value = self.explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            self.expected_value = float(np.asarray(expected_value).ravel()[-1])
        else:
            self.expected_value = float(expected_value)
        LOGGER.info("SHAP explainer initialized with %s features", len(self.feature_names))

    def explain_global(
        self,
        X_test: pd.DataFrame,
        feature_names: list[str],
        save_path: str | Path = "outputs/shap_global.png",
    ) -> None:
        if self.explainer is None or self.preprocessor is None:
            raise ValueError("Call setup() before explain_global().")
        processed = self.preprocessor.transform(X_test)
        shap_values = self.explainer.shap_values(processed)
        shap_matrix = np.asarray(shap_values[1] if isinstance(shap_values, list) else shap_values)

        dot_path = Path(save_path)
        bar_path = dot_path.with_name(f"{dot_path.stem}_bar{dot_path.suffix}")
        plt.figure()
        shap.summary_plot(shap_matrix, processed, feature_names=feature_names, plot_type="dot", show=False)
        plt.tight_layout()
        plt.savefig(dot_path, bbox_inches="tight", dpi=200)
        plt.close()

        plt.figure()
        shap.summary_plot(shap_matrix, processed, feature_names=feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(bar_path, bbox_inches="tight", dpi=200)
        plt.close()

        mean_abs = np.abs(shap_matrix).mean(axis=0)
        top_indices = np.argsort(mean_abs)[::-1][:5]
        top_features = [(feature_names[idx], float(mean_abs[idx])) for idx in top_indices]
        LOGGER.info("Top 5 SHAP features: %s", top_features)

    def explain_single(self, patient_df: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, dict[str, float]]:
        if self.explainer is None or self.preprocessor is None or self.pipeline is None:
            raise ValueError("Call setup() before explain_single().")
        processed = self.preprocessor.transform(patient_df)
        shap_values = self.explainer.shap_values(processed)
        shap_vector = np.asarray(shap_values[1] if isinstance(shap_values, list) else shap_values)[0]

        feature_value_map: dict[str, Any] = {}
        for feature in patient_df.columns:
            feature_value_map[f"num__{feature}"] = patient_df.iloc[0][feature]
            feature_value_map[f"cat__{feature}"] = patient_df.iloc[0][feature]

        shap_df = pd.DataFrame(
            {
                "feature": feature_names,
                "shap_value": shap_vector,
                "feature_value": [feature_value_map.get(feature, np.nan) for feature in feature_names],
            }
        ).sort_values("shap_value", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

        rng = np.random.default_rng(self.random_state)
        numeric_cols = patient_df.select_dtypes(include=[np.number]).columns.tolist()
        bootstrap_scores: list[float] = []
        stds = self.numeric_feature_stds if self.numeric_feature_stds is not None else pd.Series(dtype=float)
        for _ in range(50):
            sample = patient_df.copy()
            for column in numeric_cols:
                sigma = 0.01 * float(stds.get(column, 1.0) or 1.0)
                sample.loc[:, column] = float(sample.iloc[0][column]) + float(rng.normal(0, sigma))
            bootstrap_scores.append(float(self.pipeline.predict_proba(sample)[0][1]))

        mean_risk = float(np.mean(bootstrap_scores))
        std_risk = float(np.std(bootstrap_scores))
        ci = {
            "mean_risk": mean_risk,
            "ci_lower": max(mean_risk - 1.96 * std_risk, 0.0),
            "ci_upper": min(mean_risk + 1.96 * std_risk, 1.0),
        }
        return shap_df, ci

    def generate_plain_text(self, shap_df: pd.DataFrame, top_n: int = 3) -> str:
        sentences: list[str] = []
        for _, row in shap_df.head(top_n).iterrows():
            value = float(row["shap_value"])
            if abs(value) > 0.10:
                magnitude = "significantly"
            elif abs(value) > 0.05:
                magnitude = "moderately"
            else:
                magnitude = "slightly"
            direction = "increases" if value > 0 else "reduces"
            label = self.FEATURE_DESCRIPTIONS.get(row["feature"].replace("num__", "").replace("cat__", ""), (row["feature"], ""))[0]
            sentences.append(f"Your {label} {magnitude} {direction} your cardiovascular risk.")
        return " ".join(sentences)

    def what_if_delta(
        self,
        original_df: pd.DataFrame,
        modified_df: pd.DataFrame,
        pipeline: Pipeline,
        feature_names: list[str],
    ) -> dict[str, Any]:
        original_shap, _ = self.explain_single(original_df, feature_names)
        modified_shap, _ = self.explain_single(modified_df, feature_names)
        risk_original = float(pipeline.predict_proba(original_df)[0][1])
        risk_modified = float(pipeline.predict_proba(modified_df)[0][1])
        delta = risk_modified - risk_original
        summary = (
            f"Scenario changes your predicted risk from {risk_original * 100:.1f}% "
            f"to {risk_modified * 100:.1f}% ({delta * 100:+.1f} percentage points)."
        )
        return {
            "original_risk": risk_original,
            "modified_risk": risk_modified,
            "risk_delta": delta,
            "original_shap": original_shap,
            "modified_shap": modified_shap,
            "summary": summary,
        }

    def plot_waterfall(self, shap_df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
        plot_df = shap_df.head(top_n).copy().iloc[::-1]
        colors = ["#c0392b" if value > 0 else "#1e8449" for value in plot_df["shap_value"]]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(plot_df["feature"], plot_df["shap_value"], color=colors)
        ax.set_title("Top Feature Contributions")
        ax.set_xlabel("SHAP Value")
        fig.tight_layout()
        return fig


def generate_explanation(top_features: Iterable[str]) -> list[str]:
    mapping = {
        "systolic_bp": "High systolic blood pressure is the main reason your risk is elevated.",
        "bmi": "Higher BMI increases pressure on your cardiovascular system.",
        "cholesterol_raw": "Higher cholesterol adds moderate cardiovascular risk.",
        "age_years": "Age modestly increases your baseline cardiovascular risk.",
        "pulse_pressure": "A wider pulse pressure suggests added vascular strain.",
        "lifestyle_risk_score": "Lifestyle factors moderately increase your risk.",
        "age_bp_interaction": "Age and blood pressure together amplify your risk.",
        "bp_bmi_interaction": "Blood pressure and BMI together increase your risk further.",
        "glucose_raw": "Higher glucose adds metabolic risk.",
    }
    explanations: list[str] = []
    for feature in top_features:
        clean_feature = str(feature).replace("num__", "").replace("cat__", "")
        if clean_feature in mapping and mapping[clean_feature] not in explanations:
            explanations.append(mapping[clean_feature])
    return explanations


def explain_scenario(name: str) -> str:
    explanations = {
        "BP -10": "Reducing BP lowers arterial pressure and cardiovascular strain.",
        "BP -20": "A larger BP reduction can help, though model effects may be non-linear.",
        "BMI -3": "Lower BMI reduces metabolic load and can improve hemodynamics.",
        "Quit Smoking": "Smoking cessation reduces long-term vascular damage.",
        "Become Active": "Physical activity supports cardiovascular fitness and metabolic health.",
        "Original": "This is your current baseline risk profile.",
    }
    return explanations.get(name, "")


def generate_feature_impact_summary(shap_df: pd.DataFrame, top_n: int = 3) -> list[str]:
    bullets: list[str] = []
    for _, row in shap_df.head(top_n).iterrows():
        feature = str(row["feature"]).replace("num__", "").replace("cat__", "").replace("_", " ")
        direction = "increases" if float(row["shap_value"]) > 0 else "reduces"
        magnitude = abs(float(row["shap_value"]))
        qualifier = "significantly" if magnitude > 0.10 else "moderately" if magnitude > 0.05 else "slightly"
        bullets.append(f"{feature.title()} {qualifier} {direction} your predicted risk")
    return bullets


def generate_report(user_risk: float, best_action: Any) -> str:
    if best_action is None:
        best_text = "No significant improvement detected"
    elif hasattr(best_action, "get"):
        best_text = str(best_action.get("scenario", best_action))
    else:
        best_text = str(best_action)

    return (
        "Cardiovascular Risk Report\n\n"
        f"Risk: {round(user_risk * 100, 2)}%\n"
        f"Best Action: {best_text}\n\n"
        "Generated by the cardiovascular risk intelligence platform."
    )


if __name__ == "__main__":
    LOGGER.info("Run run_pipeline.py to fit the model before using SHAPExplainer.")
