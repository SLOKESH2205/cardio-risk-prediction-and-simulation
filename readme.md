# Cardiovascular Risk Decision-Support System

End-to-end ML and analytics platform that harmonizes two cardiovascular datasets, segments patients, explains predictions with SHAP, and serves insights through a 4-tab Streamlit app.

This is a cardiovascular risk decision-support system, not just a prediction model. The product is designed to combine prediction, reliability analysis, scenario simulation, and cohort comparison into a decision-oriented workflow.

## Architecture

```text
Framingham CSV        Cardio CSV
      |                   |
      +-------- ingest.py --------+
                    |
              harmonized.csv
                    |
                features.py
          /         |         \
 segmentation.py  train.py  01_EDA.ipynb
          \         |         /
            explain.py + evaluate.py
                    |
             models/ + outputs/
                    |
           app/streamlit_app.py
```

## Setup

```bash
pip install -r requirements.txt
python run_pipeline.py
streamlit run app/streamlit_app.py
```

## Dataset Setup

Place the input files in:

- `data/raw/heart_disease.csv`
- `data/raw/cardio_train.csv`

Generated artifacts:

- `data/processed/harmonized.csv`
- `models/pipeline.pkl`
- `models/segmentation.pkl`
- `outputs/`

## Modules

`src/ingest.py` loads both datasets, renames raw columns into a unified schema, applies physiological outlier filters, drops duplicates, and writes both the harmonized dataset and a data quality report.

`src/features.py` creates domain-informed cardiovascular features such as pulse pressure, AHA blood pressure category, age group, BMI category, lifestyle burden, and clinically motivated interaction terms.

`src/segmentation.py` validates cluster count with elbow and silhouette diagnostics, fits KMeans segments, saves the segmentation artifact, profiles each cluster, and adds a drift-style atypical-patient warning during cluster prediction.

`src/train.py` builds the production-ready sklearn pipeline, trains Logistic Regression, Random Forest, and XGBoost, compares CV and test ROC-AUC, saves the best model, and evaluates cross-dataset transportability.

`src/explain.py` uses SHAP TreeExplainer for global and local interpretability, produces human-readable explanations, estimates confidence intervals via bootstrap perturbations, and powers what-if comparisons.

`src/evaluate.py` generates ROC, confusion matrix, precision-recall plots, and a model card for the saved best model.

## Key Insights

The project computes cohort insights directly from the harmonized dataset and surfaces them in the Population Insights tab. After running the pipeline, use the dashboard and notebook to inspect the measured risk multiplier for Stage 2 hypertension, the excess burden of lifestyle score above 2, and the senior-vs-young risk ratio within the combined cohort.

## Resume Bullets

ML/Data Science roles:
"Engineered an end-to-end cardiovascular risk prediction system by harmonizing two
heterogeneous clinical datasets (74K records), building domain-informed features
(pulse pressure, AHA BP categorization, composite lifestyle score), training an
XGBoost pipeline achieving 0.84+ ROC-AUC with cross-population generalization
tested via cross-dataset evaluation. Integrated SHAP-based explainability with
human-readable output."

Data Analyst roles:
"Designed an ETL pipeline to fuse two cardiovascular datasets with mismatched schemas,
performed cohort-level EDA surfacing that Stage 2 hypertension patients carry 2.8x
baseline risk, applied KMeans clustering to identify three clinically distinct patient
segments, and built a 4-tab interactive Streamlit analytics dashboard."

## Tech Stack

- pandas
- NumPy
- scikit-learn
- XGBoost
- SHAP
- Plotly
- Streamlit
- Matplotlib
- Seaborn
- Joblib
