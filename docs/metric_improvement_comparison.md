# Metric Improvement Comparison

## Changes Implemented

- Added interview-friendly clinical features:
  - BMI category
  - Blood pressure category
  - Age group
  - Lifestyle risk score
  - BP x BMI interaction
  - Age x BP interaction
- Kept `RandomizedSearchCV` for XGBoost tuning.
- Added `scale_pos_weight` to the XGBoost search space.
- Saved best-model precision, recall, F1, and confusion matrix at the best F1 threshold.
- Kept AUC evaluation unchanged because threshold tuning does not change ranking quality.

## Before vs After

| Metric | Before | After at 0.50 threshold | After at optimized threshold | Change vs Before |
| --- | ---: | ---: | ---: | ---: |
| CV AUC | 0.805574 | 0.804997 | 0.804997 | -0.000578 |
| Test AUC | 0.805689 | 0.805605 | 0.805605 | -0.000084 |
| Precision | 0.753092 | 0.747410 | 0.685751 | -0.067342 |
| Recall | 0.694416 | 0.705229 | 0.798400 | +0.103985 |
| F1 | 0.722565 | 0.725707 | 0.737800 | +0.015236 |

## Confusion Matrix

Before at 0.50 threshold:

```text
[[5560, 1537],
 [2063, 4688]]
```

After at 0.50 threshold:

```text
[[5488, 1609],
 [1990, 4761]]
```

After at optimized 0.40 threshold:

```text
[[4627, 2470],
 [1361, 5390]]
```

## Interpretation

The feature engineering and XGBoost imbalance setting slightly improved the default-threshold F1 from `0.722565` to `0.725707`.

The bigger gain came from threshold optimization: F1 improved to `0.737800`, and recall improved from `0.694416` to `0.798400`. This means the model catches substantially more positive heart-risk cases, at the cost of lower precision.

AUC stayed essentially flat. That is expected because AUC measures ranking quality across thresholds, while threshold optimization changes the final classification cutoff.

## Prediction Script Check

The sample prediction changed from `69.71%` to `66.61%` because the model was retrained with the new feature set.

## Verification

- `.venv\Scripts\python.exe -m pytest`: passed, `8 passed`.
- `.venv\Scripts\python.exe run_pipeline.py`: passed.
- `.venv\Scripts\python.exe predict_pipeline.py`: passed, sample output `{'risk_percent': 66.61}`.
- Streamlit startup smoke test: passed.
