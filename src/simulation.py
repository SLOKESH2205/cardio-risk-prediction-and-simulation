"""Simulation helpers for scenarios, uncertainty, and segment narratives."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features import FeatureEngineer
from src.train import ModelTrainer


_FEATURE_ENGINEER = FeatureEngineer()
_TRAINER = ModelTrainer()


SCENARIO_MAP = {
    "Original": None,
    "Quit Smoking": "quit_smoking",
    "Become Active": "become_active",
    "BP -10": "bp_minus_10",
    "BP -20": "bp_minus_20",
    "BMI -3": "bmi_minus_3",
}


SCENARIO_EXPLANATIONS = {
    "Quit Smoking": "Why this works: smoking drives vascular inflammation and long-term cardiovascular strain.",
    "Become Active": "Why this works: physical activity improves metabolic health and overall cardiovascular resilience.",
    "BP -10": "Why this works: reducing systolic BP lowers arterial pressure and directly reduces cardiovascular strain.",
    "BP -20": "Why this works: a larger BP reduction can help, though model interactions may make the benefit non-linear.",
    "BMI -3": "Why this works: lower BMI often improves blood pressure, metabolic stress, and long-term cardiovascular burden.",
}


def apply_scenario(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    modified = df.copy()

    if scenario == "quit_smoking":
        modified["smoke"] = 0
    elif scenario == "become_active":
        modified["active"] = 1
    elif scenario == "bp_minus_10":
        modified["systolic_bp"] = modified["systolic_bp"] - 10
    elif scenario == "bp_minus_20":
        modified["systolic_bp"] = modified["systolic_bp"] - 20
    elif scenario == "bmi_minus_3":
        modified["bmi"] = modified["bmi"] - 3

    return modified


def get_prediction(pipeline: Any, df: pd.DataFrame) -> float:
    engineered = _FEATURE_ENGINEER.engineer(df.copy())
    model_features = _TRAINER.NUMERICAL_FEATURES + _TRAINER.CATEGORICAL_FEATURES
    return float(pipeline.predict_proba(engineered[model_features])[0][1])


def estimate_uncertainty(
    pipeline: Any,
    df: pd.DataFrame,
    n_samples: int = 10,
    random_state: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(random_state)
    probs: list[float] = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for _ in range(n_samples):
        sample = df.copy()
        for col in numeric_cols:
            base = float(sample.iloc[0][col])
            sigma = max(abs(base) * 0.01, 0.01)
            sample.loc[:, col] = base + float(rng.normal(0, sigma))
        probs.append(get_prediction(pipeline, sample))
    return {
        "mean": float(np.mean(probs)),
        "std": float(np.std(probs)),
    }


def get_uncertainty(
    pipeline: Any,
    df: pd.DataFrame,
    n_iter: int = 10,
    random_state: int = 42,
) -> tuple[float, float]:
    uncertainty = estimate_uncertainty(
        pipeline=pipeline,
        df=df,
        n_samples=n_iter,
        random_state=random_state,
    )
    return uncertainty["mean"], uncertainty["std"]


def run_scenarios(pipeline: Any, base_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    results: list[dict[str, Any]] = []
    base_risk = get_prediction(pipeline, base_df)

    for name, key in SCENARIO_MAP.items():
        if key is None:
            risk = base_risk
        else:
            modified = apply_scenario(base_df, key)
            risk = get_prediction(pipeline, modified)

        results.append(
            {
                "scenario": name,
                "scenario_key": key,
                "risk": round(risk * 100, 2),
                "delta": round((risk - base_risk) * 100, 2),
                "tier": _risk_tier(risk),
                "explanation": SCENARIO_EXPLANATIONS.get(name, ""),
            }
        )

    return pd.DataFrame(results), base_risk


def get_best_action(df: pd.DataFrame) -> pd.Series | None:
    improvements = df[df["delta"] < 0]
    if improvements.empty:
        return None
    return improvements.loc[improvements["delta"].idxmin()]


def get_priority_actions(df: pd.DataFrame) -> pd.DataFrame:
    improvements = df[df["delta"] < 0].sort_values("delta")
    return improvements[["scenario", "delta"]].head(3).copy()


def generate_segment_story(cluster_id: int, cluster_profiles: dict[int, dict[str, Any]]) -> str:
    profile = cluster_profiles.get(cluster_id, {})
    label = profile.get("label", f"Cluster {cluster_id}")
    description = profile.get("description", "")
    return (
        f"You belong to the '{label}' group. "
        f"Typical characteristics: {description} "
        f"This group can show rising risk if lifestyle and blood pressure are not improved."
    )


def build_report_text(
    base_risk: float,
    percentile: float,
    best_action: pd.Series | None,
    scenario_df: pd.DataFrame,
    segment_story: str,
    feature_bullets: list[str],
) -> str:
    best_line = (
        f"Best action: {best_action['scenario']} ({best_action['delta']} pts)"
        if best_action is not None
        else "Best action: No significant improvement detected"
    )
    scenario_lines = "\n".join(
        f"- {row.scenario}: {row.risk}% ({row.delta:+.2f} pts)"
        for row in scenario_df.itertuples(index=False)
    )
    feature_lines = "\n".join(f"- {item}" for item in feature_bullets)
    return (
        f"Cardiovascular Risk Report\n"
        f"Base risk: {base_risk * 100:.2f}%\n"
        f"Riskier than {percentile:.1f}% of similar individuals\n"
        f"{best_line}\n\n"
        f"Scenario breakdown:\n{scenario_lines}\n\n"
        f"Segment story:\n{segment_story}\n\n"
        f"Key risk drivers:\n{feature_lines}\n"
    )


def _risk_tier(probability: float) -> str:
    if probability < 0.30:
        return "Low Risk"
    if probability <= 0.60:
        return "Moderate Risk"
    return "High Risk"
