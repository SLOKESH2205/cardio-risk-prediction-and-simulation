# Cardiovascular Risk Intelligence Platform

A production-style healthcare AI platform for cardiovascular risk prediction, explainability, simulation, and population analytics. The system combines robust machine learning, calibration, uncertainty estimation, explainability, and decision-support workflows into a single end-to-end experience.

## What the platform delivers

- Predictive risk scoring for cardiovascular events
- Explainable outputs using SHAP and human-readable reasoning
- Trust and calibration analysis for safer decision support
- What-if intervention simulation for lifestyle and clinical scenarios
- Population analytics and clustering for risk segmentation
- Cross-dataset validation to assess generalization
- Data quality and fairness monitoring

## Architecture

```text
Raw data
  ↓
Data ingestion & quality checks
  ↓
Feature engineering (clinical features + interactions)
  ↓
Segmentation & population profiling
  ↓
Model training, tuning, calibration, threshold selection
  ↓
Explainability, uncertainty analysis, simulation
  ↓
Streamlit dashboard & report generation
```

## Key engineering improvements

- Refactored modeling stack around reusable training workflows
- Added richer healthcare-inspired features such as pulse pressure, mean arterial pressure, hypertension category, interaction terms, and polynomial features
- Introduced hyperparameter tuning, calibration assessment, threshold optimization, fairness reporting, and imbalance analysis
- Added a modular reporting layer for dashboards and future PDF/CSV exports
- Preserved the existing Streamlit experience while expanding the UI with richer insights

## Core capabilities

### Prediction
- Multiple model families: Logistic Regression, Random Forest, XGBoost, Gradient Boosting, HistGradientBoosting, LightGBM, CatBoost, Voting Ensemble, and Stacking-style ensembles
- Automated model comparison and best-model selection
- Calibration and threshold optimization for clinically meaningful decision thresholds

### Explainability
- SHAP-based global and patient-level analysis
- Human-readable clinical explanations
- Feature-attribution summaries for trust and transparency

### Simulation
- Interventions such as smoking cessation, exercise increase, blood pressure reduction, weight loss, and combined scenarios
- Scenario ranking and delta-based recommendations

### Population analytics
- Risk slicing by age group, blood pressure category, and other segments
- Surveillance of cluster-level risk patterns and data quality signals

## Run locally

```bash
python -m pip install -r requirements.txt
python run_pipeline.py
streamlit run app.py
```

## Project structure

```text
app/                  # Streamlit UI assets
data/                 # Raw and processed datasets
models/               # Saved artifacts
outputs/              # Evaluation, calibration, and reporting outputs
src/                  # Core package modules
tests/                # Regression tests
config.py             # Project-wide configuration
run_pipeline.py       # End-to-end execution entry point
```

## Notes

This platform is intended for analytics, decision support, and portfolio demonstration. It is not a medical diagnostic system and should be interpreted with clinical oversight.
