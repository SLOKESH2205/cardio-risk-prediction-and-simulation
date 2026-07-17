# Model Improvement Report

## Files modified
- src/ingest.py
- src/features.py
- src/train.py
- run_pipeline.py
- requirements.txt
- tests/test_features.py
- tests/test_prediction.py
- tests/test_training.py

## Why each modification was made
- Data cleaning now removes rows where diastolic blood pressure is greater than or equal to systolic blood pressure.
- Feature engineering now keeps only the approved simple clinical features.
- Training now compares only Logistic Regression, Random Forest, and XGBoost.
- XGBoost tuning uses RandomizedSearchCV on training folds only.
- Numeric preprocessing imputes medians and clips train-fold outliers with 1st/99th percentile winsorization.
- XGBoost uses scale_pos_weight to account for class imbalance without synthetic samples.
- Model selection now uses validation F1 instead of AUC alone.
- Threshold optimization now chooses the cutoff on validation and reports final test performance separately.
- Calibration diagnostics compare the base model with sigmoid-calibrated probabilities.
- Feature ablation retrains one-feature-drop variants and records retained features.
- Preprocessing ablation compares no clipping, 1st/99th percentile winsorization, and 5th/95th percentile clipping.
- Learning-curve and error-analysis files are saved for interview diagnostics.
- Threshold analysis, feature importance, hyperparameters, and this report are saved as assignment outputs.

## Metric comparison

metric | baseline | new | delta
--- | ---: | ---: | ---:
cv_auc | 0.8036 | 0.8032 | -0.0004
test_auc | 0.8091 | 0.8032 | -0.0060
precision | 0.7582 | 0.6573 | -0.1010
recall | 0.6810 | 0.8329 | +0.1519
f1 | 0.7175 | 0.7347 | +0.0172

## Whether each modification improved metrics
- Best current model: XGBoost
- Validation AUC changed by -0.0004.
- Test F1 changed by +0.0172.

## Best XGBoost parameters
{
  "subsample": 0.8,
  "scale_pos_weight": 1.577,
  "reg_lambda": 1.5,
  "reg_alpha": 0.05,
  "n_estimators": 300,
  "min_child_weight": 5,
  "max_depth": 3,
  "learning_rate": 0.02,
  "gamma": 0.5,
  "colsample_bytree": 0.8,
  "base_scale_pos_weight": 1.051291547078974,
  "cv_f1": 0.735578152272354,
  "cv_recall": 0.7918708750117857
}

## Best threshold
- 0.46

## Interview explanation of threshold tuning
- The validation-selected threshold did not improve test F1, which means the default 0.50 cutoff was at least as strong on this final split.

## Calibration diagnostics
{
  "method": "sigmoid",
  "selected_for_deployment": false,
  "reason": "Calibration is reported as a reliability diagnostic; the saved pipeline stays directly explainable for SHAP and feature importance.",
  "base_validation_auc": 0.8099110276179666,
  "calibrated_validation_auc": 0.809999641240395,
  "base_best_f1": 0.7426608880529888,
  "calibrated_best_f1": 0.7428379400151223,
  "base_best_threshold": 0.4599999999999999,
  "calibrated_best_threshold": 0.36999999999999994
}

## Feature ablation
- See outputs/feature_ablation.csv and outputs/selected_features.json.

## Preprocessing ablation
- See outputs/preprocessing_ablation.csv.

## Learning curve and errors
- See outputs/learning_curve.csv and outputs/learning_curve_diagnostics.json.
- See outputs/error_analysis_summary.csv, outputs/error_feature_contrasts.csv, outputs/top_false_positives.csv, and outputs/top_false_negatives.csv.

## Top 10 important features

| Feature | Importance |
| --- | ---: |
| cat__bp_category | 0.257063 |
| num__systolic_bp | 0.217180 |
| num__systolic_bp_squared | 0.116494 |
| cat__age_group | 0.108054 |
| num__age_bp_interaction | 0.061275 |
| num__alcohol_missing | 0.051480 |
| num__bp_bmi_interaction | 0.029682 |
| num__active_missing | 0.028065 |
| num__cholesterol_raw | 0.025995 |
| num__age_cholesterol_interaction | 0.022951 |

## Candidate models

| Model | CV AUC | Test AUC | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 0.8032 | 0.8098 | 0.6908 | 0.7957 | 0.7396 |
| LogisticRegression | 0.7962 | 0.8052 | 0.7323 | 0.7280 | 0.7301 |
| RandomForest | 0.7417 | 0.7453 | 0.6777 | 0.6930 | 0.6852 |

## Remaining limitations
- This is an educational risk prediction model, not a medical diagnostic tool.
- The two datasets use different feature definitions, so external validation would still be needed before real clinical use.
- Threshold tuning was evaluated after the model was fixed; it should be rechecked for any new dataset.
