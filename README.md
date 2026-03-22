# 🫀 Cardiovascular Risk Intelligence System

---

## TL;DR

An end-to-end cardiovascular risk intelligence system that combines **risk prediction, patient segmentation, explainability, and intervention simulation** to help identify high-risk individuals and recommend actionable improvements.

---

## 📊 Project Scale

* 70K+ patients (Cardio dataset)
* 4K+ patients (Framingham dataset)
* Multi-dataset training + validation
* Full pipeline: **raw data → insights → decisions**

---

## 🎯 Problem Statement

Most healthcare ML projects stop at:

> “Will this patient have heart disease?”

That is not enough.

Real-world systems must answer:

* Which patients are high-risk groups?
* What factors are driving risk?
* How reliable is the model across populations?
* What actions will reduce risk the most?

---

## 💡 Solution Overview

This project builds a **Cardiovascular Risk Intelligence System** integrating:

* Risk Prediction (ML)
* Patient Segmentation (Clustering)
* Explainability (SHAP + reasoning)
* What-if Simulation (Intervention modeling)
* Cross-dataset validation (robustness check)

👉 Transforming ML predictions into **decision intelligence**

---

## 🧠 Key Capabilities

### 1. Risk Prediction

* Predicts cardiovascular risk probability
* Models used:

  * Logistic Regression
  * Random Forest
  * XGBoost (selected)

Outputs:

* Risk %
* Risk category (Low / Moderate / High)
* Prediction confidence

---

### 2. Patient Segmentation (Personas)

Clusters patients using:

* Blood pressure
* BMI
* Cholesterol
* Lifestyle indicators

Produces interpretable groups:

* High-risk metabolic group
* Lifestyle-risk group
* Controlled/low-risk group

---

### 3. Risk Composition

Breaks prediction into components:

* Blood pressure contribution
* BMI contribution
* Lifestyle contribution
* Age interaction effects

👉 Answers: **“Why is this patient at risk?”**

---

### 4. Explainability

* SHAP-based feature importance
* Global + individual explanations
* Human-readable insights

---

### 5. What-If Simulation (🔥 Key Feature)

Simulates real interventions:

* BP reduction (-10 / -20 mmHg)
* BMI reduction
* Smoking cessation
* Increased activity

Outputs:

* Risk change (%)
* Best action recommendation
* Impact ranking

👉 Converts ML into **decision support system**

---

### 6. Risk Trajectory

* Simulates future risk progression with age
* Captures non-linear behavior

---

### 7. Cross-Dataset Generalization (🔥 Strong Point)

Train on one dataset → test on another:

* Captures real-world deployment challenges
* Highlights population shift

---

### 8. Bias & Reliability Layer

* Detects unstable features:

  * Smoking
  * Blood pressure
  * Lifestyle score

* Flags contradictions:

  * Example: smoking reducing risk due to dataset bias

👉 Adds **trust-awareness to predictions**

---

## 📈 Model Performance & Validation

### 🔹 Primary Model: XGBoost

| Metric               | Score |
| -------------------- | ----- |
| Cross-Validation AUC | 0.804 |
| Test AUC             | 0.809 |
| Precision            | 0.758 |
| Recall               | 0.681 |
| F1 Score             | 0.718 |

👉 Strong discrimination with balanced precision-recall tradeoff

---

### 🔹 Model Comparison

| Model               | CV AUC | Test AUC | Precision | Recall | F1     |
| ------------------- | ------ | -------- | --------- | ------ | ------ |
| XGBoost             | 0.8036 | 0.8091   | 0.7582    | 0.681  | 0.7175 |
| Logistic Regression | 0.7714 | 0.7768   | 0.7079    | 0.7043 | 0.7061 |
| Random Forest       | 0.7441 | 0.7421   | 0.6784    | 0.6777 | 0.6781 |

👉 XGBoost selected for best overall performance

---

### 🔹 Cross-Dataset Generalization

| Train → Test        | AUC   |
| ------------------- | ----- |
| Cardio → Framingham | 0.664 |
| Framingham → Cardio | 0.685 |

👉 Shows realistic performance drop due to population differences

---

### 🔹 Segmentation Quality

* 3 distinct patient clusters
* Meaningful separation of risk groups

---

### 🔹 Key Observations

* Strong in-dataset performance (~0.80 AUC)
* Cross-dataset drop highlights real-world limitations
* BP, BMI, and lifestyle interactions dominate risk
* Some features show instability → flagged

---

### 🔹 What This Means

* ✔ Good predictive power
* ✔ Realistic evaluation (not overfitted)
* ✔ Includes robustness + bias awareness
* ✔ Demonstrates production thinking

---

## 🖥️ Dashboard Features

* Risk Predictor
* Population Insights
* Model Report
* What-if Analysis

Interactive capabilities:

* Patient input sliders
* Risk breakdown
* Cluster comparison
* Scenario simulation
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

## ⚙️ Technical Design

### Preprocessing

* Missing value handling
* Feature alignment across datasets
* BMI calculation
* BP categorization

---

### Feature Engineering

* Pulse pressure
* Age-BP interaction
* Lifestyle risk score
* BMI-BP interaction

---

### Modeling

* Multi-model training
* Cross-validation
* Model selection

---

### Simulation Engine

* Scenario-based feature modification
* Delta risk calculation
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

## ▶️ How to Run

```
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
├── src/                # Core ML modules
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

## 💡 Example Insights

* Blood pressure is the strongest risk driver
* BMI amplifies cardiovascular stress
* Lifestyle factors interact with clinical features
* Risk behavior is non-linear
* Dataset bias can affect interpretation

---

## ⚠️ Limitations

* Not a medical diagnostic system
* Dataset bias affects some features
* Cross-population generalization is limited
* Observational data → not causal inference

---

## 🎯 Use Cases

* Healthcare analytics
* Preventive screening tools
* Risk awareness platforms
* ML explainability demonstrations
* Portfolio (ML + Data + Product thinking)

---

## 🚀 Why This Project Stands Out

This is not just a prediction model.

It:

* Combines **prediction + segmentation + simulation**
* Explains *why* predictions happen
* Converts outputs into **actionable decisions**
* Evaluates real-world reliability (cross-dataset)
* Handles bias and uncertainty

👉 Designed as a **decision intelligence system**, not just ML

---

## 👨‍💻 Author

Lokesh S

---

## ⭐ Final Note

This project demonstrates how ML evolves from:

👉 *Prediction → Insight → Decision → Impact*

---

⭐ If you found this useful, consider starring the repo!
