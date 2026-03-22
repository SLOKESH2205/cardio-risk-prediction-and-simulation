"""Streamlit interface for the cardiovascular risk decision-support platform."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analysis import (
    build_decision_summary,
    check_outlier,
    detect_drift,
    feature_stability,
    get_percentile,
    project_risk_trajectory,
    risk_breakdown,
    scenario_display_name,
    trust_score,
)
from src.explainability import SHAPExplainer, generate_explanation, generate_report
from src.segmentation import PatientSegmenter, describe_cluster
from src.simulation import (
    build_report_text,
    generate_segment_story,
    get_best_action,
    get_priority_actions,
    get_uncertainty,
    run_scenarios,
)
from src.features import FeatureEngineer
from src.train import ModelTrainer
from src.utils import load_joblib, load_json


BASE_DIR = Path(__file__).resolve().parents[0]
if BASE_DIR.name == "app":
    BASE_DIR = BASE_DIR.parent

MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "harmonized.csv"
RISK_REFERENCE_PATH = OUTPUTS_DIR / "risk_reference.csv"
CLUSTER_PROFILES_PATH = OUTPUTS_DIR / "cluster_profiles.csv"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .app-card {
            background-color: #111827;
            padding: 18px 20px;
            border-radius: 14px;
            margin-bottom: 14px;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .app-card h3 {
            margin: 0;
            color: #f9fafb;
            font-size: 1.08rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def card(title: str) -> None:
    st.markdown(
        f"""
        <div class="app-card">
            <h3>{title}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )


def compact_stat(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div style="padding:4px 0 10px 0;">
            <div style="font-size:0.95rem;color:#d1d5db;margin-bottom:6px;">{label}</div>
            <div style="font-size:1.4rem;font-weight:600;line-height:1.2;color:#f9fafb;word-break:break-word;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str) -> None:
    st.markdown(f"## {title}")
    st.markdown("---")


def plotly_style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def get_missing_artifacts() -> list[Path]:
    required = [
        MODELS_DIR / "pipeline.pkl",
        MODELS_DIR / "segmentation.pkl",
        PROCESSED_PATH,
        RISK_REFERENCE_PATH,
        CLUSTER_PROFILES_PATH,
    ]
    return [path for path in required if not path.exists()]


@st.cache_resource
def load_assets() -> tuple[object, dict, pd.DataFrame, SHAPExplainer, list[str], dict[int, dict], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pipeline = load_joblib(MODELS_DIR / "pipeline.pkl")
    segmentation_artifact = load_joblib(MODELS_DIR / "segmentation.pkl")
    harmonized = pd.read_csv(PROCESSED_PATH)
    featured = FeatureEngineer().engineer(harmonized)
    trainer = ModelTrainer(BASE_DIR)
    feature_names = trainer.get_feature_names_from_pipeline(pipeline)
    explainer = SHAPExplainer(BASE_DIR)
    explainer.setup(pipeline, featured[trainer.NUMERICAL_FEATURES + trainer.CATEGORICAL_FEATURES])
    cluster_profiles = segmentation_artifact.get("cluster_profiles", {})
    risk_reference = pd.read_csv(RISK_REFERENCE_PATH)
    cluster_profiles_df = pd.read_csv(CLUSTER_PROFILES_PATH)
    stability_df = feature_stability(featured)
    return pipeline, segmentation_artifact, featured, explainer, feature_names, cluster_profiles, risk_reference, cluster_profiles_df, stability_df


def initialize_session_state() -> None:
    defaults = {
        "age": 52,
        "gender": "Male",
        "height": 170,
        "weight": 75,
        "systolic_bp": 135,
        "diastolic_bp": 85,
        "cholesterol_raw": 220,
        "glucose_raw": 120,
        "smoke": False,
        "alcohol": False,
        "active": True,
        "last_patient_features": None,
        "last_prediction": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def risk_tier(probability: float) -> str:
    if probability < 0.30:
        return "Low Risk"
    if probability <= 0.60:
        return "Moderate Risk"
    return "High Risk"


def bp_category_label(encoded_value: int) -> str:
    return {0: "Normal", 1: "Elevated", 2: "Stage 1", 3: "Stage 2"}.get(int(encoded_value), "Unknown")


def build_patient_dataframe() -> pd.DataFrame:
    bmi = st.session_state["weight"] / ((st.session_state["height"] / 100) ** 2)
    return pd.DataFrame(
        [{
            "age_years": int(st.session_state["age"]),
            "gender_bin": 1 if st.session_state["gender"] == "Male" else 0,
            "systolic_bp": float(st.session_state["systolic_bp"]),
            "diastolic_bp": float(st.session_state["diastolic_bp"]),
            "bmi": float(bmi),
            "cholesterol_raw": float(st.session_state["cholesterol_raw"]),
            "glucose_raw": float(st.session_state["glucose_raw"]),
            "smoke": int(st.session_state["smoke"]),
            "alcohol": int(st.session_state["alcohol"]),
            "active": int(st.session_state["active"]),
            "target": 0,
            "source": "app",
        }]
    )


def percentile_context(probability: float, risk_reference: pd.DataFrame) -> float:
    return get_percentile(probability, risk_reference)


def _model_features() -> list[str]:
    trainer = ModelTrainer(BASE_DIR)
    return trainer.NUMERICAL_FEATURES + trainer.CATEGORICAL_FEATURES


def _drift_flag(value: float, series: pd.Series) -> bool:
    mean_value = float(series.mean())
    std_value = float(series.std())
    if std_value == 0:
        return False
    z_value = (float(value) - mean_value) / std_value
    return detect_drift(float(z_value), 0.0)


def _cluster_comparison(cluster_profiles_df: pd.DataFrame, cluster_id: int, user_bp: float, user_bmi: float, user_risk: float) -> dict[str, float] | None:
    if cluster_profiles_df.empty:
        return None
    match = cluster_profiles_df.loc[cluster_profiles_df["cluster"] == cluster_id]
    if match.empty:
        return None
    row = match.iloc[0]
    return {
        "user_bp": float(user_bp),
        "cluster_bp": float(row["systolic_bp_mean"]),
        "user_bmi": float(user_bmi),
        "cluster_bmi": float(row["bmi_mean"]),
        "user_risk": float(user_risk * 100),
        "cluster_risk": float(row["target_rate"] * 100),
    }


def _format_signed(value: float) -> str:
    return f"{float(value):+.2f}"


def _format_pct(value: float) -> str:
    return f"{float(value):.2f}%"


def _trajectory_is_non_monotonic(trajectory_df: pd.DataFrame) -> bool:
    return bool((trajectory_df["risk"].diff().fillna(0) < 0).any())


def _scenario_interpretation(name: str, delta: float) -> str:
    if name == "Quit Smoking" and delta > 0:
        return "Model anomaly"
    if delta <= -10:
        return "Strong improvement"
    if delta <= -5:
        return "Moderate improvement"
    if delta < 0:
        return "Minor improvement"
    if delta > 0:
        return "Risk increase"
    return "No material change"


def _clinical_meaning(breakdown_df: pd.DataFrame) -> str:
    top_components = breakdown_df.head(2)["component"].tolist()
    if "Blood Pressure" in top_components and "Body Composition" in top_components:
        return "Your risk is driven more by pressure-related and metabolic factors than lifestyle alone."
    if "Blood Pressure" in top_components:
        return "Blood pressure is the clearest driver of your current risk profile."
    if "Lifestyle" in top_components:
        return "Lifestyle contributes meaningfully, but it is not the only driver of your risk."
    return "Your risk reflects interacting cardiovascular factors rather than a single isolated input."


def render_tab1(
    pipeline: object,
    explainer: SHAPExplainer,
    feature_names: list[str],
    cluster_profiles: dict[int, dict],
    featured_df: pd.DataFrame,
    risk_reference: pd.DataFrame,
    cluster_profiles_df: pd.DataFrame,
    stability_df: pd.DataFrame,
) -> None:
    section_header("Section 1: Overview")
    input_col, output_col = st.columns([1, 1.25])
    predict_clicked = False

    with input_col:
        with st.container():
            card("Patient Inputs")
            st.session_state["age"] = st.slider("Age", 18, 90, st.session_state["age"])
            st.session_state["gender"] = st.radio("Gender", ["Male", "Female"], index=0 if st.session_state["gender"] == "Male" else 1)
            st.session_state["height"] = st.slider("Height (cm)", 140, 210, st.session_state["height"])
            st.session_state["weight"] = st.slider("Weight (kg)", 40, 150, st.session_state["weight"])
            bmi = st.session_state["weight"] / ((st.session_state["height"] / 100) ** 2)
            st.caption(f"Computed BMI: {bmi:.2f}")
            st.session_state["systolic_bp"] = st.slider("Systolic BP", 80, 200, st.session_state["systolic_bp"])
            st.session_state["diastolic_bp"] = st.slider("Diastolic BP", 50, 130, st.session_state["diastolic_bp"])
            tmp_patient = FeatureEngineer().engineer(build_patient_dataframe())
            st.caption(f"Pulse pressure: {float(tmp_patient.iloc[0]['pulse_pressure']):.2f} mmHg | BP category: {bp_category_label(int(tmp_patient.iloc[0]['bp_category']))}")
            st.session_state["cholesterol_raw"] = st.selectbox("Cholesterol", [150, 220, 280], index=[150, 220, 280].index(st.session_state["cholesterol_raw"]), format_func=lambda x: {150: "Normal (150)", 220: "Above normal (220)", 280: "Well above normal (280)"}[x])
            st.session_state["glucose_raw"] = st.selectbox("Glucose", [80, 120, 180], index=[80, 120, 180].index(st.session_state["glucose_raw"]), format_func=lambda x: {80: "Normal (80)", 120: "Above normal (120)", 180: "Well above normal (180)"}[x])
            st.session_state["smoke"] = st.checkbox("Smoke", value=st.session_state["smoke"])
            st.session_state["alcohol"] = st.checkbox("Alcohol", value=st.session_state["alcohol"])
            st.session_state["active"] = st.checkbox("Active", value=st.session_state["active"])
            lifestyle = (int(st.session_state["smoke"]) * 2) + int(st.session_state["alcohol"]) + (1 - int(st.session_state["active"]))
            st.caption(f"Lifestyle risk score: {lifestyle}/4")
            predict_col, reset_col = st.columns(2)
            predict_clicked = predict_col.button("Predict Risk", use_container_width=True)
            if reset_col.button("Reset patient", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

    with output_col:
        if predict_clicked:
            patient_df_raw = build_patient_dataframe()
            patient_df = FeatureEngineer().engineer(patient_df_raw)
            model_features = _model_features()
            probability = float(pipeline.predict_proba(patient_df[model_features])[0][1])
            tier = risk_tier(probability)
            shap_df, ci = explainer.explain_single(patient_df[model_features], feature_names)
            cluster_id, cluster_label = PatientSegmenter(BASE_DIR).predict_cluster(patient_df)
            _uncertainty_mean, uncertainty_std = get_uncertainty(pipeline, patient_df_raw)
            percentile = percentile_context(probability, risk_reference)
            results_df, _ = run_scenarios(pipeline, patient_df_raw)
            best_action = get_best_action(results_df)
            cluster_description = describe_cluster(cluster_id)
            top_features = shap_df.head(3)["feature"].tolist()
            feature_explanations = generate_explanation(top_features)
            trust_level, trust_reason = trust_score(top_features, stability_df)
            breakdown_df = risk_breakdown(shap_df)
            trajectory_df = project_risk_trajectory(pipeline, patient_df_raw)
            cluster_comparison = _cluster_comparison(cluster_profiles_df, cluster_id, float(patient_df_raw.iloc[0]["systolic_bp"]), float(patient_df_raw.iloc[0]["bmi"]), probability)
            decision_summary = build_decision_summary(probability, tier, feature_explanations, best_action, trust_level, trust_reason)
            st.session_state["last_patient_features"] = patient_df_raw
            st.session_state["last_prediction"] = {
                "probability": probability,
                "ci": ci,
                "uncertainty_std": uncertainty_std,
                "cluster_id": cluster_id,
                "cluster_label": cluster_label,
                "cluster_description": cluster_description,
                "percentile": percentile,
                "feature_explanations": feature_explanations,
                "trust_level": trust_level,
                "trust_reason": trust_reason,
                "breakdown_df": breakdown_df,
                "trajectory_df": trajectory_df,
                "drift_flags": {
                    "systolic_bp": _drift_flag(float(patient_df_raw.iloc[0]["systolic_bp"]), featured_df["systolic_bp"]),
                    "bmi": _drift_flag(float(patient_df_raw.iloc[0]["bmi"]), featured_df["bmi"]),
                },
                "outlier_flags": {
                    "systolic_bp": check_outlier(float(patient_df_raw.iloc[0]["systolic_bp"]), float(featured_df["systolic_bp"].mean()), float(featured_df["systolic_bp"].std())),
                    "bmi": check_outlier(float(patient_df_raw.iloc[0]["bmi"]), float(featured_df["bmi"].mean()), float(featured_df["bmi"].std())),
                },
                "best_action": best_action,
                "cluster_comparison": cluster_comparison,
                "decision_summary": decision_summary,
                "trajectory_warning": _trajectory_is_non_monotonic(trajectory_df),
                "nonlinear_warning": bool(not results_df.loc[results_df["scenario"] == "BP -10", "risk"].empty and not results_df.loc[results_df["scenario"] == "BP -20", "risk"].empty and float(results_df.loc[results_df["scenario"] == "BP -20", "risk"].iloc[0]) > float(results_df.loc[results_df["scenario"] == "BP -10", "risk"].iloc[0])),
            }
        bundle = st.session_state.get("last_prediction")
        if bundle is not None:
            probability = float(bundle["probability"])
            ci = bundle["ci"]
            percentile = float(bundle["percentile"])
            comparison = bundle.get("cluster_comparison")
            decision_summary = bundle["decision_summary"]
            breakdown_df = bundle["breakdown_df"].copy()

            with st.container():
                card("Decision Strip")
                c1, c2, c3 = st.columns(3)
                with c1:
                    compact_stat("Risk", decision_summary["risk_label"])
                with c2:
                    compact_stat("Best Action", decision_summary["top_action"])
                with c3:
                    compact_stat("Confidence", decision_summary["confidence"])
                st.info(f"Insight: {decision_summary['impact_summary']}")
                st.caption(f"95% CI: {_format_pct(ci['ci_lower'] * 100)} to {_format_pct(ci['ci_upper'] * 100)} | Higher risk than {percentile:.2f}% of similar individuals")

            with st.container():
                card("Clinical Insight Panel")
                st.markdown("**Why your risk is high:**")
                for row in breakdown_df.head(3).itertuples(index=False):
                    st.write(f"- {row.component} is contributing {_format_pct(row.contribution_pct)} of the model signal.")
                for explanation in bundle["feature_explanations"][:2]:
                    st.write(f"- {explanation}")
                st.markdown("**You vs Similar Patients:**")
                if comparison is not None:
                    st.write(f"- BP: {comparison['user_bp']:.2f} vs {comparison['cluster_bp']:.2f}")
                    st.write(f"- BMI: {comparison['user_bmi']:.2f} vs {comparison['cluster_bmi']:.2f}")
                    st.write(f"- Risk: {_format_pct(comparison['user_risk'])} vs {_format_pct(comparison['cluster_risk'])}")
                else:
                    st.write(f"- Cluster: {bundle['cluster_label']}")
                st.markdown("**What this means:**")
                st.write(_clinical_meaning(breakdown_df))
                st.caption(f"Assigned group: {bundle['cluster_label']} | {bundle['cluster_description']}")
                st.caption(f"Confidence reason: {bundle['trust_reason']}")
                if bundle["drift_flags"]["systolic_bp"] or bundle["drift_flags"]["bmi"]:
                    st.warning("One or more inputs differ noticeably from the training distribution.")
                if bundle["outlier_flags"]["systolic_bp"]:
                    st.warning("Outlier warning: your systolic BP is significantly higher than typical dataset values.")
                if bundle["outlier_flags"]["bmi"]:
                    st.warning("Outlier warning: your BMI is significantly different from typical dataset values.")
                if bundle["nonlinear_warning"]:
                    st.warning("Model behavior insight: risk reduction is not always linear due to feature interactions.")

            vis_col, trajectory_col = st.columns(2)
            with vis_col:
                with st.container():
                    card("Risk Composition")
                    breakdown_fig = px.bar(
                        breakdown_df,
                        x="contribution_pct",
                        y="component",
                        orientation="h",
                        color="contribution_pct",
                        color_continuous_scale="Blues",
                        title="Risk Composition by Component",
                    )
                    st.plotly_chart(plotly_style(breakdown_fig), use_container_width=True)

            with trajectory_col:
                with st.container():
                    card("Projected Risk Trajectory")
                    trajectory_df = bundle["trajectory_df"].copy()
                    trajectory_df["risk_pct"] = trajectory_df["risk"] * 100
                    traj_fig = px.line(trajectory_df, x="age_years", y="risk_pct", markers=True, title="Projected Risk with Age Progression")
                    st.plotly_chart(plotly_style(traj_fig), use_container_width=True)
                    for row in trajectory_df.itertuples(index=False):
                        st.write(f"- Age {int(row.age_years)} -> {_format_pct(row.risk * 100)}")
                    if bundle["trajectory_warning"]:
                        st.warning("Projection is non-monotonic due to model interactions and should be interpreted cautiously.")


def render_tab2(featured_df: pd.DataFrame) -> None:
    section_header("Population Insights")
    filter_cols = st.columns(4)
    age_values = sorted(featured_df["age_group"].dropna().unique())
    bp_values = sorted(featured_df["bp_category"].dropna().unique())
    selected_age = filter_cols[0].multiselect("Age Group", age_values, default=age_values)
    selected_gender = filter_cols[1].radio("Gender", ["All", "Male", "Female"], horizontal=True)
    selected_bp = filter_cols[2].multiselect("BP Category", bp_values, default=bp_values)
    selected_source = filter_cols[3].radio("Source", ["All", "framingham", "cardio"], horizontal=True)

    filtered = featured_df[featured_df["age_group"].isin(selected_age) & featured_df["bp_category"].isin(selected_bp)].copy()
    if selected_gender != "All":
        filtered = filtered[filtered["gender_bin"] == (1 if selected_gender == "Male" else 0)]
    if selected_source != "All":
        filtered = filtered[filtered["source"] == selected_source]

    metrics_cols = st.columns(4)
    metrics_cols[0].metric("Total patients", f"{len(filtered):,}")
    metrics_cols[1].metric("Cardio-positive rate", _format_pct(filtered['target'].mean() * 100))
    metrics_cols[2].metric("Mean age", f"{filtered['age_years'].mean():.2f}")
    metrics_cols[3].metric("Mean systolic BP", f"{filtered['systolic_bp'].mean():.2f}")

    with st.container():
        card("Risk by Group")
        age_fig = plotly_style(px.bar(filtered.groupby("age_group", as_index=False)["target"].mean(), x="age_group", y="target", title="Risk Rate by Age Group"))
        bp_fig = plotly_style(px.bar(filtered.groupby("bp_category", as_index=False)["target"].mean(), x="bp_category", y="target", color="bp_category", title="Risk Rate by BP Category"))
        left, right = st.columns(2)
        left.plotly_chart(age_fig, use_container_width=True)
        right.plotly_chart(bp_fig, use_container_width=True)

    with st.container():
        card("Lifestyle and Correlation")
        lifestyle_fig = plotly_style(px.histogram(filtered, x="lifestyle_risk_score", color="target", marginal="box", barmode="overlay", title="Lifestyle Risk Distribution"))
        corr_cols = ["age_years", "bmi", "systolic_bp", "diastolic_bp", "pulse_pressure", "cholesterol_raw", "glucose_raw", "lifestyle_risk_score", "target"]
        corr_fig = plotly_style(px.imshow(filtered[corr_cols].corr(numeric_only=True), text_auto=".2f", title="Correlation Heatmap"))
        left, right = st.columns(2)
        left.plotly_chart(lifestyle_fig, use_container_width=True)
        right.plotly_chart(corr_fig, use_container_width=True)

    normal_risk = float(filtered.loc[filtered["bp_category"] == 0, "target"].mean()) if (filtered["bp_category"] == 0).any() else 0.0
    stage2_risk = float(filtered.loc[filtered["bp_category"] == 3, "target"].mean()) if (filtered["bp_category"] == 3).any() else 0.0
    high_life = float(filtered.loc[filtered["lifestyle_risk_score"] > 2, "target"].mean()) if (filtered["lifestyle_risk_score"] > 2).any() else 0.0
    low_life = float(filtered.loc[filtered["lifestyle_risk_score"] <= 2, "target"].mean()) if (filtered["lifestyle_risk_score"] <= 2).any() else 0.0
    young_risk = float(filtered.loc[filtered["age_group"] == 0, "target"].mean()) if (filtered["age_group"] == 0).any() else 0.0
    senior_risk = float(filtered.loc[filtered["age_group"] == 2, "target"].mean()) if (filtered["age_group"] == 2).any() else 0.0

    with st.container():
        card("Insight Summary")
        st.info(f"Patients in Stage 2 hypertension have {(stage2_risk / normal_risk if normal_risk else 0.0):.2f}x the risk of Normal BP patients.")
        st.info(f"Lifestyle score above 2 is associated with {(((high_life - low_life) / low_life * 100) if low_life else 0.0):.2f}% higher risk.")
        st.info(f"Senior patients (60+) have {(senior_risk / young_risk if young_risk else 0.0):.2f}x the risk of Young Adults.")


def render_tab3(stability_df: pd.DataFrame) -> None:
    section_header("Model Report")
    comparison_path = OUTPUTS_DIR / "model_comparison.csv"
    cross_path = OUTPUTS_DIR / "cross_dataset_eval.csv"
    model_card_path = OUTPUTS_DIR / "model_card.json"
    data_quality_path = OUTPUTS_DIR / "data_quality_report.json"

    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        best_row = comparison.sort_values("test_auc", ascending=False).iloc[0]
        with st.container():
            card("Model Comparison")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Selected Model", str(best_row["model"]))
            c2.metric("CV AUC", f"{best_row['cv_auc']:.3f}")
            c3.metric("Test AUC", f"{best_row['test_auc']:.3f}")
            c4.metric("F1", f"{best_row['f1']:.3f}")
            display_comparison = comparison[["model", "cv_auc", "test_auc", "precision", "recall", "f1"]].copy()
            display_comparison.columns = ["Model", "CV AUC", "Test AUC", "Precision", "Recall", "F1"]
            st.dataframe(display_comparison, use_container_width=True)
            st.info("XGBoost was selected because it achieved the best AUC and handles non-linear feature interactions better than Logistic Regression and Random Forest in this mixed cohort setting.")

    roc_col, curve_col = st.columns([1, 2])
    with roc_col:
        card("Cross-Dataset Generalization")
        if cross_path.exists():
            cross_df = pd.read_csv(cross_path)
            cardio_auc = float(cross_df.loc[cross_df["direction"] == "cardio_to_framingham", "auc"].iloc[0]) if (cross_df["direction"] == "cardio_to_framingham").any() else 0.0
            frame_auc = float(cross_df.loc[cross_df["direction"] == "framingham_to_cardio", "auc"].iloc[0]) if (cross_df["direction"] == "framingham_to_cardio").any() else 0.0
            gap = ((max(cardio_auc, frame_auc) - min(cardio_auc, frame_auc)) / max(cardio_auc, frame_auc) * 100) if max(cardio_auc, frame_auc) else 0.0
            st.metric("Cardio -> Framingham AUC", f"{cardio_auc:.3f}")
            st.metric("Framingham -> Cardio AUC", f"{frame_auc:.3f}")
            st.markdown(f"### Cross-Dataset Insight\n\n- Cardio -> Framingham: AUC = {cardio_auc:.3f}\n- Framingham -> Cardio: AUC = {frame_auc:.3f}\n\nPerformance shifts by about {gap:.2f}% across datasets, which justifies caution on unstable features.")

    with curve_col:
        st.markdown("### ROC Curve Analysis")
        roc_path = OUTPUTS_DIR / "roc_curve.png"
        if roc_path.exists():
            st.image(str(roc_path), use_container_width=True)
        st.info("These technical diagnostics are intentionally kept in the Model Report tab, separate from the decision-support view.")

    cm_col, shap_col = st.columns(2)
    with cm_col:
        st.markdown("### Confusion Matrix")
        cm_path = OUTPUTS_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
    with shap_col:
        st.markdown("### Global SHAP Importance")
        shap_path = OUTPUTS_DIR / "shap_global.png"
        if shap_path.exists():
            st.image(str(shap_path), use_container_width=True)

    with st.container():
        card("Feature Stability Across Datasets")
        display_df = stability_df[["feature_label", "framingham_corr", "cardio_corr", "status"]].copy()
        display_df["framingham_corr"] = display_df["framingham_corr"].map(_format_signed)
        display_df["cardio_corr"] = display_df["cardio_corr"].map(_format_signed)
        display_df.columns = ["Feature", "Framingham", "Cardio", "Stability"]
        st.dataframe(display_df, use_container_width=True)
        unstable = display_df.loc[display_df["Stability"] == "unstable", "Feature"].tolist()
        if unstable:
            st.warning(f"Unstable features detected across datasets: {', '.join(unstable)}")

    with st.container():
        card("Data Quality")
        if data_quality_path.exists():
            data_quality = load_json(data_quality_path)
            cardio_removed = data_quality.get("rows_removed", {}).get("cardio", {})
            framingham_removed = data_quality.get("rows_removed", {}).get("framingham", {})
            st.markdown("**Removed from cardio dataset:**")
            st.write(f"- {cardio_removed.get('systolic_bp', 0)} invalid systolic BP values")
            st.write(f"- {cardio_removed.get('diastolic_bp', 0)} diastolic anomalies")
            st.write(f"- {cardio_removed.get('duplicates', 0)} duplicates")
            st.markdown("**Removed from Framingham dataset:**")
            st.write(f"- {framingham_removed.get('systolic_bp', 0)} invalid systolic BP values")
            st.write(f"- {framingham_removed.get('bmi', 0)} BMI anomalies")
            with st.expander("Full Data Quality Report"):
                st.json(data_quality)

    with st.container():
        card("Limitations")
        st.warning("This is a cardiovascular risk decision-support system, not a diagnostic system.")
        st.warning("Some predictions may reflect dataset bias and unstable feature behavior.")
        st.warning("Model performance shifts across cohorts, so transportability is limited.")
        if model_card_path.exists():
            model_card = load_json(model_card_path)
            for item in model_card.get("known_limitations", []):
                st.warning(item)


def render_tab4(pipeline: object, cluster_profiles: dict[int, dict], risk_reference: pd.DataFrame) -> None:
    if st.session_state.get("last_patient_features") is None:
        st.info("Please run a prediction in Tab 1 first.")
        return

    base_df = st.session_state["last_patient_features"].copy()
    results_df, base_risk = run_scenarios(pipeline, base_df)
    best = get_best_action(results_df)
    priority_actions = get_priority_actions(results_df)
    percentile = percentile_context(base_risk, risk_reference)
    current_bundle = st.session_state["last_prediction"]
    segment_story = generate_segment_story(int(current_bundle["cluster_id"]), cluster_profiles)

    section_header("What-if Analysis")

    with st.container():
        card("Scenario Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Base Risk", _format_pct(base_risk * 100))
        c2.metric("Risk Category", risk_tier(base_risk))
        c3.metric("Percentile", f"{percentile:.2f}%")
        if best is not None:
            st.success(f"Best scenario: {scenario_display_name(str(best['scenario']))} changes risk by {float(best['delta']):+.2f}%.")

    scenario_rows = results_df.copy()
    scenario_rows = scenario_rows[scenario_rows["scenario"] != "Original"].copy()
    scenario_rows["Scenario"] = scenario_rows["scenario"].map(scenario_display_name)
    scenario_rows["Risk"] = scenario_rows["risk"].map(_format_pct)
    scenario_rows["Change"] = scenario_rows["delta"].apply(lambda value: f"{'Down ' if value < 0 else 'Up ' if value > 0 else 'Flat '}{abs(float(value)):.2f}")
    scenario_rows["Interpretation"] = scenario_rows.apply(lambda row: _scenario_interpretation(str(row["scenario"]), float(row["delta"])), axis=1)
    scenario_rows = scenario_rows.sort_values("delta")

    with st.container():
        card("Scenario Table")
        st.dataframe(scenario_rows[["Scenario", "Risk", "Change", "Interpretation"]], use_container_width=True)
        if "Quit Smoking" in results_df["scenario"].values:
            smoking_row = results_df[results_df["scenario"] == "Quit Smoking"].iloc[0]
            if float(smoking_row["delta"]) > 0:
                st.warning("Quit Smoking is flagged as a model anomaly here. Clinically, smoking increases cardiovascular risk.")
        bp_minus_10 = results_df.loc[results_df["scenario"] == "BP -10", "risk"]
        bp_minus_20 = results_df.loc[results_df["scenario"] == "BP -20", "risk"]
        if not bp_minus_10.empty and not bp_minus_20.empty and float(bp_minus_20.iloc[0]) > float(bp_minus_10.iloc[0]):
            st.warning("Risk reduction is not always linear due to feature interactions.")

    with st.container():
        card("Recommended Actions")
        if priority_actions.empty:
            st.warning("No improving actions were detected for the current profile.")
        else:
            for rank, row in enumerate(priority_actions.itertuples(index=False), start=1):
                st.write(f"{rank}. {scenario_display_name(str(row.scenario))} -> improves risk by {abs(float(row.delta)):.2f}%")

    with st.container():
        card("Scenario Details")
        detail_rows = results_df.copy().sort_values("delta")
        detail_rows["display_name"] = detail_rows["scenario"].map(scenario_display_name)
        detail_rows["interpretation"] = detail_rows.apply(lambda row: _scenario_interpretation(str(row["scenario"]), float(row["delta"])), axis=1)
        for row in detail_rows.itertuples(index=False):
            if str(row.scenario) == "Original":
                continue
            with st.expander(str(row.display_name)):
                st.write(f"Risk: {_format_pct(row.risk)}")
                st.write(f"Change: {float(row.delta):+.2f}%")
                st.write(f"Interpretation: {row.interpretation}")
                if hasattr(row, "explanation") and str(row.explanation):
                    st.caption(str(row.explanation))

    with st.container():
        card("Interpretation")
        st.info(f"This is a cardiovascular risk decision-support system. Base risk is {_format_pct(base_risk * 100)}, and the strongest simulated lever is {scenario_display_name(str(best['scenario'])) if best is not None else 'not available'}.")
        if best is not None:
            st.write(f"- Best action: {scenario_display_name(str(best['scenario']))} changes risk by {float(best['delta']):+.2f}%")
        st.write(f"- Percentile context: higher risk than {percentile:.2f}% of similar individuals")
        st.write(segment_story)
        st.caption(f"Assigned group: {current_bundle['cluster_label']} | {current_bundle['cluster_description']}")

    with st.container():
        card("Export Report")
        summary_report = generate_report(base_risk, best)
        detailed_report = build_report_text(base_risk=base_risk, percentile=percentile, best_action=best, scenario_df=results_df, segment_story=segment_story, feature_bullets=current_bundle.get("feature_explanations", []))
        report_text = f"{summary_report}\n\n{detailed_report}"
        st.download_button("Download Report", report_text, file_name="cardio_risk_report.txt")


def render_missing_artifacts_message(missing_paths: list[Path]) -> None:
    st.title("Cardiovascular Risk Decision-Support System")
    st.caption("Prediction, segmentation, simulation, and reliability analysis")
    st.error("Required trained artifacts are missing, so the app cannot load yet.")
    st.code("python run_pipeline.py")
    for path in missing_paths:
        st.write(f"- `{path}`")


def main() -> None:
    st.set_page_config(page_title="Cardiovascular Risk Decision-Support System", layout="wide")
    inject_styles()
    initialize_session_state()

    missing_paths = get_missing_artifacts()
    if missing_paths:
        render_missing_artifacts_message(missing_paths)
        st.stop()

    pipeline, segmentation_artifact, featured_df, explainer, feature_names, cluster_profiles, risk_reference, cluster_profiles_df, stability_df = load_assets()
    _ = segmentation_artifact
    st.title("Cardiovascular Risk Decision-Support System")
    st.caption("This is a cardiovascular risk decision-support system, not just a prediction model.")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Risk Predictor", "Population Insights", "Model Report", "What-If Analysis"])
    with tab1:
        render_tab1(pipeline, explainer, feature_names, cluster_profiles, featured_df, risk_reference, cluster_profiles_df, stability_df)
    with tab2:
        render_tab2(featured_df)
    with tab3:
        render_tab3(stability_df)
    with tab4:
        render_tab4(pipeline, cluster_profiles, risk_reference)


if __name__ == "__main__":
    main()











