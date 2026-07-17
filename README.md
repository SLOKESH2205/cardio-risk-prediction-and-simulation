# 🫀 Cardiovascular Risk Intelligence Platform

**An end-to-end ML decision-support system for cardiovascular risk — prediction, explainability, patient segmentation, what-if simulation, and honest cross-population validation.**

[![Live Demo](https://img.shields.io/badge/demo-streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://cardio-risk-prediction-and-simulation.streamlit.app)
[![Python](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Model](https://img.shields.io/badge/model-XGBoost-006400)](#-model-performance)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#-testing)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#-license)

🌐 **Live demo:** https://cardio-risk-prediction-and-simulation.streamlit.app
📦 **Repo:** https://github.com/SLOKESH2205/cardio-risk-prediction-and-simulation

---

## Table of Contents

- [TL;DR](#tldr)
- [Problem Statement](#-problem-statement)
- [Solution Overview](#-solution-overview)
- [Project Scale](#-project-scale)
- [Core Features](#-core-features)
- [Model Performance](#-model-performance)
- [Feature Importance & Ablations](#-feature-importance--ablations)
- [Cross-Dataset Generalization](#-cross-dataset-generalization)
- [Patient Segmentation](#-patient-segmentation)
- [Data Quality](#-data-quality)
- [Dashboard Tour](#️-dashboard-tour)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Getting Started](#️-getting-started)
- [Project Structure](#-project-structure)
- [Testing](#-testing)
- [Limitations](#️-limitations)
- [Use Cases](#-use-cases)
- [Roadmap](#-roadmap)
- [Author](#-author)

---

## TL;DR

Most ML heart-disease projects stop at *"will this patient have heart disease?"*. This one goes further, combining **risk scoring, explainability, patient segmentation, what-if simulation, and cross-population validation** into a single pipeline + dashboard — so it answers not just *who* is at risk, but *why*, *how confidently*, and *what to do about it*.

## 🎯 Problem Statement

A prediction alone doesn't support a decision. A useful clinical-analytics system needs to answer four questions at once:

| Question | Capability that answers it |
| --- | --- |
| Who is at risk? | XGBoost risk classifier + calibrated probability |
| Why is this patient at risk? | SHAP global + per-patient explainability |
| Can we trust this prediction? | Calibration diagnostics, threshold analysis, error analysis |
| What action reduces risk the most? | What-if intervention simulator |
| Does this generalize beyond one population? | Cross-dataset validation (train on one cohort, test on another) |

## 💡 Solution Overview

```
Risk Prediction  →  Explainability  →  Simulation  →  Segmentation  →  Validation  →  Decision
```

The platform integrates six pillars on top of a shared, harmonized dataset:

1. **Risk Prediction** — XGBoost classifier, tuned via `RandomizedSearchCV`
2. **Explainability** — SHAP-based global and per-patient attributions
3. **What-If Simulation** — models the risk impact of BP reduction, BMI reduction, smoking cessation, and activity increase
4. **Patient Segmentation** — KMeans clustering into clinically distinct personas
5. **Threshold & Calibration Diagnostics** — decision threshold chosen on a validation split (never on test), with calibration reported as a trust diagnostic
6. **Cross-Dataset Validation** — train on one cohort, evaluate on the other, to quantify (not hide) the real-world generalization gap

## 📊 Project Scale

- **69,238** harmonized patient records, merged from two independent cohorts (Kaggle **Cardio Train** dataset and the **Framingham Heart Study** dataset)
- Base positive (heart-disease) rate across the harmonized cohort: **48.75%**
- Mean patient age **52.6 years**, mean systolic BP **127.3 mmHg**
- **22 model-ready features** after cleaning, harmonization, and engineering
- Full reproducible pipeline: raw data → cleaning → features → model → explainability → decisions, with every metric backed by a saved artifact under `outputs/`

## 🧠 Core Features

### 1. Risk Prediction
- Three model families compared head-to-head: **XGBoost**, Logistic Regression, Random Forest (see [Model Performance](#-model-performance))
- Final model: **XGBoost**, selected on validation F1
- Outputs a calibration-checked risk %, a Low / Moderate / High risk category, and the contributing factors
- Decision threshold selected **on the validation split, never on test** — final threshold: **0.46**

### 2. Explainability (SHAP)
- Global feature-importance ranking, cross-checked against independent feature-ablation experiments (not just raw SHAP magnitude)
- Per-patient explanations so a prediction is never a black box
- Answers *"why this prediction?"* at both the population and individual level

### 3. What-If Simulation 🔥
- Simulates targeted interventions: blood-pressure reduction, BMI reduction, smoking cessation, increased physical activity
- Reports the resulting risk delta per intervention and ranks interventions by impact
- Turns a static score into an actionable care-planning tool

### 4. Patient Segmentation (KMeans)
Three behaviorally and clinically distinct personas emerge from clustering on demographic, metabolic, and lifestyle features (see [Patient Segmentation](#-patient-segmentation) for the full profile table).

### 5. Threshold & Calibration Diagnostics
- Precision/recall tradeoff deliberately tuned for a **screening use case** — high recall is preferred over high precision, since missing a true at-risk patient is costlier than a false alarm
- Calibration (Platt/sigmoid scaling) is computed and reported, but **not** used for deployment — the raw XGBoost output is kept so SHAP explanations stay directly attributable to the deployed model
- Every threshold decision is backed by a saved sweep (`outputs/threshold_sweep.csv`) and a validation-vs-test comparison (`outputs/threshold_analysis.json`)

### 6. Cross-Dataset Validation 🔥
Train on one cohort, test on the other, to measure how much performance is lost when the model meets a population it has never seen — see [Cross-Dataset Generalization](#-cross-dataset-generalization).

### 7. Bias & Reliability Layer
- **Learning-curve diagnosis**: validation F1 barely moves as more training data is added (a gain of **+0.00013 F1** from smallest to full training set), which points to a **feature/data information ceiling**, not overfitting or an undersized dataset
- **Feature ablation**: every engineered feature was tested by one-at-a-time removal and retraining; results in [Feature Importance & Ablations](#-feature-importance--ablations)
- **Preprocessing ablation**: outlier handling strategies (none / 1–99 percentile winsorizing / 5–95 percentile clipping) were compared, and "no additional clipping" won on validation F1
- **Error analysis**: top false positives, false negatives, and their feature contrasts are saved for inspection (`outputs/top_false_positives.csv`, `outputs/top_false_negatives.csv`, `outputs/error_feature_contrasts.csv`)

## 📈 Model Performance

### Final model: XGBoost (test set, decision threshold = 0.46)

| Metric | Score |
| --- | --- |
| Test ROC-AUC | **0.803** |
| CV AUC (5-fold) | 0.803 |
| Precision | 0.657 |
| Recall | 0.833 |
| F1 Score | 0.735 |
| Confusion matrix (TN, FP / FN, TP) | `[[4165, 2932], [1128, 5623]]` |

### Model comparison (validation-selected threshold)

| Model | CV AUC | Test AUC | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **XGBoost** ✅ | 0.803 | 0.810 | 0.691 | 0.796 | **0.740** |
| Logistic Regression | 0.796 | 0.805 | 0.732 | 0.728 | 0.730 |
| Random Forest | 0.742 | 0.745 | 0.678 | 0.693 | 0.685 |

### The precision/recall tradeoff, explained

| Metric | Default 0.50 threshold | Deployed 0.46 threshold | Change |
| --- | ---: | ---: | ---: |
| Test AUC | 0.803 | 0.803 | ≈ 0 (AUC is threshold-independent) |
| Precision | 0.685 | 0.657 | −0.028 |
| Recall | 0.795 | 0.833 | **+0.038** |
| F1 | 0.736 | 0.735 | ≈ flat |

AUC sits at a realistic ceiling of **~0.80–0.81** for this feature set — the learning-curve and ablation diagnostics both confirm this is a *signal* ceiling, not a tuning shortfall. The deliberate design choice is to trade a modest amount of precision for materially higher recall, because in a **screening** context, missing a true positive (a false negative) is more costly than an extra follow-up test (a false positive).

> On the deployed threshold, the model produces **2,932 false positives** and **1,128 false negatives** out of ~13,848 held-out test patients. Both error populations, and the features that most distinguish them from correct predictions, are saved as CSVs for audit.

## 🔎 Feature Importance & Ablations

Top predictive features (XGBoost gain-based importance):

| Rank | Feature | Importance |
| --- | --- | ---: |
| 1 | Blood-pressure category (engineered) | 0.257 |
| 2 | Systolic BP | 0.217 |
| 3 | Systolic BP² (engineered) | 0.116 |
| 4 | Age group (engineered) | 0.108 |
| 5 | Age × BP interaction (engineered) | 0.061 |
| 6 | Alcohol-missingness indicator | 0.051 |
| 7 | BP × BMI interaction (engineered) | 0.030 |
| 8 | Activity-missingness indicator | 0.028 |
| 9 | Cholesterol | 0.026 |
| 10 | Age × cholesterol interaction (engineered) | 0.023 |

Blood pressure — both raw and engineered — dominates the model, which matches clinical intuition around hypertension as a primary cardiovascular risk driver.

**Feature ablation** (drop-one-feature-and-retrain, 22 candidate features tested): every engineered feature was validated this way rather than assumed useful. The largest single-feature validation-F1 swing across all 12 ablation variants was **≈ ±0.0005**, meaning no individual engineered feature is doing outsized, fragile work — the model's performance is distributed and robust to any one feature's removal.

**Preprocessing ablation** (outlier handling): "no additional clipping" (`none`) beat both 1–99 percentile winsorizing and 5–95 percentile clipping on validation F1 (0.7427 vs 0.7423 vs 0.7422) — a small but consistent edge, so the simplest preprocessing strategy was kept.

**Learning-curve diagnosis**: `data_or_feature_bottleneck`. Final train F1 (0.739) and final validation F1 (0.737) are nearly identical (**generalization gap of 0.0018**), and validation F1 improves only **+0.00013** from the smallest to the full training set. This rules out overfitting and rules out "just add more data" — the ceiling is the available feature signal.

## 🌍 Cross-Dataset Generalization

Train on one cohort, test on the other — a stricter test than a random train/test split from the same population:

| Train → Test | AUC |
| --- | ---: |
| Cardio → Framingham | 0.664 |
| Framingham → Cardio | 0.685 |

Same-population AUC is **~0.80–0.81**; cross-population AUC drops to **~0.66–0.69**. That ~0.13–0.15 AUC gap is reported deliberately, rather than hidden behind an inflated single-split number — it quantifies the real limit on how far this model's conclusions transport across genuinely different patient populations (the two source studies differ in era, geography, and available fields — Framingham lacks alcohol-use and physical-activity fields that the Cardio dataset has).

## 👥 Patient Segmentation

KMeans clustering (on demographic, metabolic, blood-pressure, and lifestyle features) surfaces three personas:

| Persona | Share of patients | Heart-disease rate | Mean age | Mean BMI | Mean systolic BP | Mean cholesterol |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 🔴 High-Risk Metabolic | 30.6% | 76.9% | 54.7 | 29.7 | 146.5 mmHg | 191 |
| 🟡 Moderate-Risk Lifestyle | 61.2% | 32.9% | 51.3 | 26.2 | 117.3 mmHg | 164 |
| 🟠 High-Risk Metabolic (elevated cholesterol) | 8.2% | 61.9% | 55.2 | 29.0 | 129.8 mmHg | **241** |

The two "high-risk" clusters are driven by different mechanisms — one by elevated blood pressure and BMI, the other primarily by markedly elevated cholesterol despite more moderate BP — which is exactly the kind of distinction a single global risk score can't surface on its own, but segmentation can.

## 🧪 Data Quality

Harmonization draws from two independent raw sources and applies deliberate, logged cleaning:

- **69,238** rows retained after cleaning and de-duplication
- Positive (heart-disease) rate: **48.7%**
- Cardio cohort cleaning removed invalid blood-pressure readings (1,236 rows), duplicate records (3,188 rows), and a smaller number of BMI/systolic/diastolic outliers
- Framingham cohort cleaning removed rows with missing glucose (388), missing cholesterol (52), and a small number of missing BMI values

Every removal reason and count is logged in `outputs/data_quality_report.md` / `.json` / `.html`, so data loss is auditable rather than silent.

## 🖥️ Dashboard Tour

| Tab | What it shows |
| --- | --- |
| 🔮 **Risk Predictor** | Patient-level risk %, risk category, and SHAP-based explanation for that specific patient |
| 🌍 **Population Insights** | Cohort risk distributions, demographic breakdowns, and data-quality summaries |
| 📊 **Model Report** | Model comparison, threshold analysis, calibration diagnostics, ROC/PR curves, confusion matrix, learning curve, feature/preprocessing ablations, cross-dataset generalization, and error analysis |
| 🎛️ **What-If Analysis** | Interactive sliders to simulate blood-pressure, BMI, smoking, and activity interventions and see the resulting risk change |

## 🧱 Architecture

```
Raw Data (Cardio + Framingham)
        ↓
Cleaning & Harmonization        (src/ingest.py)
        ↓
Feature Engineering             (src/features.py)
        ↓
Patient Segmentation (KMeans)   (src/segmentation.py)
        ↓
Model Training + Tuning         (src/train.py)   →  XGBoost selected via RandomizedSearchCV
        ↓
Threshold Selection (validation split only, no test leakage)
        ↓
Calibration Diagnostics         (reported, not deployed)
        ↓
Explainability (SHAP)           (src/explainability.py)
        ↓
What-If Simulation              (src/simulation.py)
        ↓
Cross-Dataset Validation        (src/evaluate.py)
        ↓
Streamlit Dashboard             (app/streamlit_app.py)
```

### Engineering highlights

**Feature engineering**
- Pulse pressure, mean arterial pressure
- Age×BP, BP×BMI, Age×cholesterol interaction terms
- BP / age / BMI clinical category buckets
- Composite lifestyle risk score
- Explicit missingness indicators for alcohol and activity fields (informative in the harmonized cohort)

**Modeling**
- `RandomizedSearchCV` hyperparameter tuning across XGBoost, Logistic Regression, and Random Forest
- Class-imbalance correction via XGBoost's `scale_pos_weight`
- 5-fold stratified cross-validation

**Evaluation rigor**
- Validation-only threshold selection (zero test-set leakage into the decision cutoff)
- Feature-ablation and preprocessing-ablation studies, each backed by a saved CSV
- Calibration diagnostics reported separately from the deployment decision
- Cross-dataset (cross-population) generalization testing, not just a single held-out split

## 🛠 Tech Stack

| Layer | Tools |
| --- | --- |
| Language | Python 3.11 |
| Data | Pandas, NumPy |
| Modeling | scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn, Plotly |
| App / Dashboard | Streamlit |
| Testing | pytest |

See `requirements.txt` for pinned versions.

## ▶️ Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/SLOKESH2205/cardio-risk-prediction-and-simulation.git
cd cardio-risk-prediction-and-simulation

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Run the full pipeline (ingestion → features → training → evaluation → artifacts)
python run_pipeline.py

# 4. Launch the dashboard
streamlit run app.py
```

**Quick single-patient prediction from the command line:**

```bash
python predict_pipeline.py
```

## 📂 Project Structure

```
├── app/
│   └── streamlit_app.py         # Streamlit UI (all dashboard tabs)
├── app.py                       # Compatibility launcher for the Streamlit app
├── config.py                    # Project-wide configuration/constants
├── data/
│   ├── raw/                     # Original Cardio + Framingham datasets
│   └── processed/               # Harmonized dataset used by the pipeline
├── docs/                        # Audit notes and metric-improvement history
├── models/
│   ├── pipeline.pkl             # Trained end-to-end prediction pipeline
│   └── segmentation.pkl         # Trained KMeans segmentation artifact
├── notebooks/
│   └── 01_EDA.ipynb             # Exploratory data analysis
├── outputs/                     # Every metric, plot, and diagnostic artifact (source of truth)
├── predict_pipeline.py          # Sample CLI for single-patient prediction
├── run_pipeline.py              # End-to-end pipeline entry point
├── src/
│   ├── ingest.py                # Cleaning & harmonization
│   ├── features.py              # Feature engineering
│   ├── train.py                 # Training, tuning, threshold selection
│   ├── evaluate.py              # Evaluation plots, cross-dataset validation, model card
│   ├── explainability.py        # SHAP explanations
│   ├── simulation.py            # What-if intervention engine
│   ├── segmentation.py          # Patient clustering
│   ├── analysis.py              # Risk context, stability, drift helpers
│   ├── services/reporting.py    # Population-summary reporting
│   ├── utils.py                 # Shared JSON/joblib/directory helpers
│   ├── logger.py                # Shared logger factory
│   └── exception.py             # Custom ingestion exception
├── tests/                       # Regression tests (features, training, prediction, simulation)
├── requirements.txt
├── runtime.txt
└── setup.py
```

## 🧪 Testing

```bash
pytest
```

Regression tests cover feature engineering (`test_features.py`), training utilities (`test_training.py`), prediction output shape/columns (`test_prediction.py`), and the what-if simulation engine (`test_simulation.py`).

## ⚠️ Limitations

- **Not a medical diagnostic system.** This is a decision-support and analytics tool, not a substitute for clinical judgment.
- **Cross-population generalization is limited.** AUC drops from ~0.80 (same-population) to ~0.66–0.69 (cross-population) — see [Cross-Dataset Generalization](#-cross-dataset-generalization).
- **Observational data → not causal.** The what-if simulator models statistical association between risk factors and outcome, not a validated causal treatment effect; it should be read as directional decision support, not a clinical trial result.
- **AUC ceiling (~0.80–0.81) reflects available feature signal, not a tuning shortfall** — confirmed independently by the learning-curve diagnostic and feature-ablation study.
- **Harmonization asymmetry.** The Framingham cohort lacks alcohol-use and physical-activity fields present in the Cardio cohort; missingness indicators are used to make this explicit to the model rather than silently imputing.

## 🎯 Use Cases

- Healthcare analytics and preventive risk-screening workflows
- ML explainability and responsible-AI demonstrations
- Data science / ML engineering portfolio piece
- Teaching example for calibration, threshold selection, and cross-population validation done rigorously

## 🗺 Roadmap

- [ ] Add calibrated-probability toggle in the dashboard for users who want reliability-weighted outputs
- [ ] Extend cross-dataset validation to a third external cohort
- [ ] Add SHAP-based cohort drift monitoring for population shifts over time
- [ ] Export a patient-level PDF risk report from the dashboard


## 👨‍💻 Author

**Lokesh S**

---

⭐ If this project helped you, consider starring the repo!
