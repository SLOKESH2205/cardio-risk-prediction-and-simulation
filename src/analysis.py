"""Analysis helpers for risk context, stability checks, drift, and trajectories."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from src.logger import get_logger


LOGGER = get_logger(__name__)


FEATURE_LABELS = {
    "age_years": "Age",
    "systolic_bp": "Systolic BP",
    "diastolic_bp": "Diastolic BP",
    "bmi": "BMI",
    "cholesterol_raw": "Cholesterol",
    "glucose_raw": "Glucose",
    "smoke": "Smoking",
    "pulse_pressure": "Pulse Pressure",
    "lifestyle_risk_score": "Lifestyle Score",
    "bp_bmi_interaction": "BP-BMI Interaction",
    "age_bp_interaction": "Age-BP Interaction",
    "bp_category": "BP Category",
    "age_group": "Age Group",
    "bmi_category": "BMI Category",
}


RISK_BREAKDOWN_MAP = {
    "Blood Pressure": {"systolic_bp", "diastolic_bp", "pulse_pressure", "bp_category", "bp_bmi_interaction"},
    "Age Interaction": {"age_years", "age_group", "age_bp_interaction"},
    "Cholesterol": {"cholesterol_raw", "glucose_raw"},
    "Lifestyle": {"smoke", "lifestyle_risk_score"},
    "Body Composition": {"bmi", "bmi_category"},
}


SCENARIO_DISPLAY = {
    "Original": "Current Profile",
    "Quit Smoking": "Quit Smoking",
    "Become Active": "Become Active",
    "BP -10": "Reduce BP by 10 mmHg",
    "BP -20": "Reduce BP by 20 mmHg",
    "BMI -3": "Reduce BMI by 3",
}


def get_percentile(user_risk: float, population_df: pd.DataFrame) -> float:
    risk_column = "risk"
    if risk_column not in population_df.columns:
        risk_column = "risk_score"
    return float((population_df[risk_column] < user_risk).mean() * 100)


def feature_stability(df: pd.DataFrame, threshold: float = 0.20) -> pd.DataFrame:
    feature_order = [
        "systolic_bp",
        "pulse_pressure",
        "smoke",
        "cholesterol_raw",
        "glucose_raw",
        "bmi",
        "age_years",
        "lifestyle_risk_score",
    ]
    rows: list[dict[str, Any]] = []
    for feature in feature_order:
        row: dict[str, Any] = {"feature": feature, "feature_label": FEATURE_LABELS.get(feature, feature)}
        correlations: dict[str, float] = {}
        for source, source_df in df.groupby("source"):
            corr = float(source_df[[feature, "target"]].corr(numeric_only=True).loc[feature, "target"])
            correlations[str(source)] = corr
            row[f"{source}_corr"] = corr
        frame_corr = correlations.get("framingham", 0.0)
        cardio_corr = correlations.get("cardio", 0.0)
        correlation_gap = abs(frame_corr - cardio_corr)
        inconsistent_sign = frame_corr != 0.0 and cardio_corr != 0.0 and ((frame_corr > 0) != (cardio_corr > 0))
        unstable = inconsistent_sign and correlation_gap > threshold
        row["status"] = "unstable" if unstable else "stable"
        row["status_icon"] = "unstable" if unstable else "stable"
        row["correlation_gap"] = correlation_gap
        row["inconsistent_sign"] = inconsistent_sign
        rows.append(row)
    stability_df = pd.DataFrame(rows).sort_values(
        by=["status", "correlation_gap", "feature_label"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    LOGGER.info("Computed feature stability table for %s features", len(stability_df))
    return stability_df


def trust_score(important_features: Sequence[str], stability_df: pd.DataFrame) -> tuple[str, str]:
    cleaned = [str(feature).replace("num__", "").replace("cat__", "") for feature in important_features]
    unstable = set(stability_df.loc[stability_df["status"] == "unstable", "feature"].tolist())
    matched = [feature for feature in cleaned if feature in unstable]
    if not matched:
        return "High", "Top features behave consistently across Framingham and Cardio datasets."
    if cleaned and matched[0] == cleaned[0]:
        feature_name = FEATURE_LABELS.get(matched[0], matched[0])
        return "Low", f"Key feature ({feature_name}) shows inconsistent behavior across datasets."
    feature_names = ", ".join(FEATURE_LABELS.get(feature, feature) for feature in matched[:2])
    return "Medium", f"Some important drivers ({feature_names}) vary across datasets, so interpret this estimate cautiously."


def risk_breakdown(shap_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for category, features in RISK_BREAKDOWN_MAP.items():
        mask = shap_df["feature"].str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False).isin(features)
        contribution = float(shap_df.loc[mask, "shap_value"].abs().sum())
        rows.append({"component": category, "contribution": contribution})
    rows.append(
        {
            "component": "Other",
            "contribution": float(shap_df["shap_value"].abs().sum()) - sum(row["contribution"] for row in rows),
        }
    )
    breakdown_df = pd.DataFrame(rows)
    breakdown_df["contribution"] = breakdown_df["contribution"].clip(lower=0.0)
    total = float(breakdown_df["contribution"].sum())
    breakdown_df["contribution_pct"] = breakdown_df["contribution"].apply(lambda value: (value / total * 100) if total else 0.0)
    return breakdown_df.sort_values("contribution_pct", ascending=False).reset_index(drop=True)


def scenario_display_name(name: str) -> str:
    return SCENARIO_DISPLAY.get(name, name)


def build_decision_summary(
    probability: float,
    tier: str,
    feature_explanations: Sequence[str],
    best_action: pd.Series | None,
    trust_level: str,
    trust_reason: str,
) -> dict[str, Any]:
    main_drivers = list(feature_explanations[:2])
    if not main_drivers:
        main_drivers = ["Model contributions are distributed across multiple features."]
    if len(main_drivers) == 1:
        main_drivers.append("Blood pressure, age, and metabolic interactions remain relevant context.")

    if best_action is None:
        action_text = "No strong improvement was detected in the current scenario set."
        top_action = "No clear action"
    else:
        action_text = f"{scenario_display_name(str(best_action['scenario']))} ({float(best_action['delta']):+.2f}%)"
        top_action = action_text

    impact_summary = "Your risk is primarily driven by blood pressure, and reducing it offers the highest improvement potential."
    return {
        "risk_label": f"{probability * 100:.2f}% ({tier.replace(' Risk', '')})",
        "main_drivers": main_drivers,
        "best_action": action_text,
        "top_action": top_action,
        "confidence": trust_level,
        "confidence_reason": trust_reason,
        "impact_summary": impact_summary,
    }


def detect_drift(user_input: float, train_mean: float) -> bool:
    diff = abs(user_input - train_mean)
    return bool(diff > 2)


def check_outlier(input_val: float, mean: float, std: float) -> bool:
    if std == 0:
        return False
    return bool(abs(input_val - mean) > 2 * std)


def project_risk_trajectory(
    pipeline: Any,
    base_df: pd.DataFrame,
    years: Sequence[int] = (0, 5, 10),
) -> pd.DataFrame:
    from src.simulation import get_prediction

    rows: list[dict[str, Any]] = []
    base_age = int(base_df.iloc[0]["age_years"])
    for increment in years:
        scenario_df = base_df.copy()
        scenario_df.loc[:, "age_years"] = min(base_age + int(increment), 90)
        rows.append(
            {
                "age_years": int(scenario_df.iloc[0]["age_years"]),
                "risk": float(get_prediction(pipeline, scenario_df)),
            }
        )
    return pd.DataFrame(rows)

