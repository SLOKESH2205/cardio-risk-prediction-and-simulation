# 🫀 Cardiovascular Risk Intelligence Platform

🔗 **Live App:** https://cardio-risk-prediction-and-simulation.streamlit.app
📦 **Repository:** 

---

## 🚀 TL;DR

An end-to-end **ML-powered cardiovascular risk intelligence system** that goes beyond prediction — combining:

✔ Risk Prediction
✔ Patient Segmentation
✔ Explainability
✔ What-if Simulation
✔ Cross-Dataset Validation

👉 Built to transform ML outputs into **real-world decision support**

---

## 📊 Project Scale

* **70,000+ patients** (Cardio dataset)
* **4,000+ patients** (Framingham dataset)
* Multi-dataset training & validation
* Full pipeline: **Raw Data → Insights → Decisions**

---

## 🎯 Problem Statement

Most ML projects stop at:

> “Will this patient have heart disease?”

That’s incomplete.

Real systems must answer:

* Who are the high-risk groups?
* Why is the patient at risk?
* Can we trust this prediction?
* What actions reduce risk the most?

---

## 💡 Solution Overview

This project builds a **Cardiovascular Risk Intelligence Platform** integrating:

* 📌 Risk Prediction (ML)
* 📌 Patient Segmentation (Clustering)
* 📌 Explainability (SHAP + reasoning)
* 📌 What-if Simulation (Intervention modeling)
* 📌 Cross-dataset validation (robustness)

👉 Moving from **prediction → decision intelligence**

---

## 🧠 Core Features

### 1. Risk Prediction

* Models: Logistic Regression, Random Forest, **XGBoost (selected)**
* Outputs:

  * Risk %
  * Risk category (Low / Moderate / High)
  * Confidence score

---

### 2. Patient Segmentation

Clustering based on:

* Blood Pressure
* BMI
* Cholesterol
* Lifestyle indicators

Produces:

* High-risk metabolic group
* Lifestyle-risk group
* Low-risk group

---

### 3. Explainability

* SHAP-based explanations
* Global + individual insights
* Human-readable reasoning

👉 Answers: **“Why this prediction?”**

---

### 4. What-If Simulation 🔥

Simulates interventions:

* BP reduction (-10 / -20 mmHg)
* BMI reduction
* Smoking cessation
* Increased activity

Outputs:

* Risk change (%)
* Best action recommendation
* Impact ranking

👉 Converts ML into **actionable healthcare insights**

---

### 5. Risk Composition

Breakdown of risk drivers:

* Blood pressure contribution
* BMI impact
* Lifestyle effect
* Feature interactions

---

### 6. Cross-Dataset Validation 🔥

Train on one dataset → test on another:

| Train → Test        | AUC   |
| ------------------- | ----- |
| Cardio → Framingham | 0.664 |
| Framingham → Cardio | 0.685 |

👉 Demonstrates **real-world generalization limits**

---

### 7. Bias & Reliability Layer

* Detects unstable features:

  * Smoking
  * Blood pressure
  * Lifestyle score

* Flags contradictions:

  * Example: smoking reducing risk (dataset bias)

👉 Adds **trust-awareness to ML predictions**

---

## 📈 Model Performance

### 🔹 Best Model: XGBoost

| Metric    | Score |
| --------- | ----- |
| CV AUC    | 0.804 |
| Test AUC  | 0.809 |
| Precision | 0.758 |
| Recall    | 0.681 |
| F1 Score  | 0.718 |

---

### 🔹 Model Comparison

| Model               | CV AUC | Test AUC | F1     |
| ------------------- | ------ | -------- | ------ |
| XGBoost             | 0.8036 | 0.8091   | 0.7175 |
| Logistic Regression | 0.7714 | 0.7768   | 0.7061 |
| Random Forest       | 0.7441 | 0.7421   | 0.6781 |

---

## 🖥️ Dashboard Features

* Risk Predictor
* Population Insights
* Model Report
* What-if Analysis

Interactive UI includes:

* Sliders for patient inputs
* Risk breakdown
* Scenario comparison
* Cluster insights
* Explainability charts

---

## 🧱 Architecture

```
Raw Data  
   ↓  
Feature Engineering  
   ↓  
Segmentation (KMeans)  
   ↓  
ML Model (XGBoost)  
   ↓  
Explainability (SHAP)  
   ↓  
Simulation Engine  
   ↓  
Streamlit Dashboard  
```

---

## ⚙️ Technical Highlights

### Feature Engineering

* Pulse pressure
* Age-BP interaction
* BMI-BP interaction
* Lifestyle risk score

### Modeling

* Cross-validation
* Multi-model comparison
* Cross-dataset evaluation

### Simulation Engine

* Scenario-based feature modification
* Risk delta computation
* Best action recommendation

---

## 🛠 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* SHAP
* Streamlit
* Plotly / Matplotlib

---

## ▶️ Run Locally

```bash
git clone https://github.com/SLOKESH2205/cardio-risk-prediction-and-simulation.git
cd cardio-risk-prediction-and-simulation

pip install -r requirements.txt
python run_pipeline.py
streamlit run app.py
```

---

## 📂 Project Structure

```
├── app/                # Streamlit UI
├── data/               # Datasets
├── models/             # Saved models
├── src/                # Core modules
│   ├── train.py
│   ├── features.py
│   ├── segmentation.py
│   ├── analysis.py
│   ├── explainability.py
│   ├── simulation.py
│   └── utils.py
│
├── app.py
├── run_pipeline.py
├── requirements.txt
└── README.md
```

---

## ⚠️ Limitations

* Not a medical diagnostic system
* Dataset bias affects some predictions
* Cross-population generalization is limited
* Observational data → not causal

---

## 🎯 Use Cases

* Healthcare analytics
* Preventive risk screening
* ML explainability demos
* Data science portfolio (ML + DA + product thinking)

---

## 🏆 Why This Project Stands Out

This is NOT a basic ML project.

It:

✔ Combines prediction + segmentation + simulation
✔ Explains model behavior
✔ Handles bias and uncertainty
✔ Validates across datasets
✔ Focuses on **decision-making, not just prediction**


---

## 👨‍💻 Author

**Lokesh S**

---

## ⭐ Final Note

This project demonstrates:

👉 **Prediction → Insight → Decision → Impact**

---

⭐ If this helped you, consider starring the repo!
