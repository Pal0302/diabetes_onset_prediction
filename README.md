# Diabetes Onset Prediction

Predicting diabetes onset from routine diagnostic measurements, using the Pima Indians
Diabetes Dataset. An end-to-end data science portfolio project — from raw data with real
quality issues, through feature engineering and model comparison, to a final model evaluated
honestly on a held-out test set.

**[Full Project Report (PDF)](https://github.com/Pal0302/diabetes_onset_prediction/blob/main/Reports/Diabetes_Prediction_Project_Report.pdf)** — problem
statement, methodology, results, and confusion matrix findings in detail.

## Problem Statement

Diabetes is a chronic condition where early detection meaningfully improves patient outcomes,
but diagnosis typically requires lab testing that isn't always immediately available. This
project explores whether a small set of routine diagnostic measurements — glucose, BMI, age,
blood pressure, insulin, and similar — can support a predictive screening signal for diabetes
risk, framed as a binary classification problem.

## Dataset

- **Source:** [Pima Indians Diabetes Dataset](https://www.kaggle.com/api/v1/datasets/download/mdssh784/pima-indians-diabetes-dataset)
- **Records:** 768 patients, 8 diagnostic features, 1 binary target (`Outcome`)
- **Class balance:** ~65% no diabetes / 35% diabetes — moderately imbalanced, which shaped
  both modeling and evaluation choices throughout

## Results

Four models were trained, tuned, and evaluated exactly once against a held-out test set of
154 patients that was never used during development:

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **KNN (final model)** | **0.747** | **0.821** | 0.653 | 0.593 | 0.621 |
| Random Forest | 0.740 | 0.818 | 0.659 | 0.537 | 0.592 |
| Logistic Regression | 0.721 | 0.795 | 0.628 | 0.500 | 0.557 |
| Decision Tree | 0.688 | 0.764 | 0.636 | 0.259 | 0.368 |

**KNN wins on both Accuracy and ROC-AUC** and is the project's final model. It correctly
identifies 59% of diabetic patients and 83% of healthy patients in the test set.

**Notable finding:** Decision Tree's Accuracy (0.688) looks only mildly worse than the other
models, but its Recall is just 0.259 — it missed roughly 3 out of every 4 diabetic patients.
Accuracy alone hides this completely; it only shows up once Precision/Recall/F1 are checked
per class, which is exactly the risk a class-imbalanced dataset like this one creates for a
screening tool.

**Error analysis on the final model:** patients KNN misses skew toward milder, more
"atypical" readings — lower average Glucose, BMI, Age, and family-history score than the
diabetic patients it catches correctly. Full breakdown in the report.

## Project Structure

```
diabetes_onset_prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── diabetes.csv
├── Final_Report.pdf
│
├── reports/
│   └── Diabetes_Prediction_Project_Report.pdf
│
├── data/
│   ├── train_cleaned.csv
│   └── test_cleaned.csv
│
├── module_1_eda_cleaning/
│   ├── module_1_eda_cleaning.ipynb
│   └── module1_eda_cleaning_report.pptx
│    
│
├── module_2_feature_engineering_modeling/
│   ├── module2_feature_engineering_modeling.ipynb
│   └── models/
│       ├── final_models.joblib
│       ├── scaler.joblib
│       └── feature_cols.joblib
│
└── module_3_evaluation/
    └── module3_model_evaluation.ipynb
    
```

## Progress

- [x] **Module 1 — Data Understanding & EDA:** cleaned the dataset, diagnosed and fixed
  hidden missing values (encoded as zeros), corrected an outlier-detection bug, explored
  feature distributions and correlations.
- [x] **Module 2 — Feature Engineering & Modeling:** compared scaling strategies, selected
  the top 5 features, tuned four models (Logistic Regression, KNN, Decision Tree, Random
  Forest) via GridSearchCV, and ran an isolated class-weighting experiment to address
  imbalance.
- [x] **Module 3 — Evaluation & Final Report:** evaluated all four tuned models once against
  the untouched test set, selected KNN as the final model, and ran a confusion matrix analysis on its
  mistakes.

## Key Findings

- Five features (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) used `0` to
  silently encode missing values — up to 48.7% missing in `Insulin`. Treated as missing data,
  not outliers, and imputed using the training-set median only (no leakage into test data).
- Caught and fixed a real bug: computing outlier bounds *after* imputation falsely inflated
  the apparent outlier count in `Insulin`, because the imputed values compressed the
  interquartile range. Recomputing bounds from originally observed values only fixed this.
- Caught and fixed a second bug in the class-weighting experiment: an early version let
  Decision Tree's `max_depth` silently revert to unconstrained while adding
  `class_weight='balanced'`, producing a misleading result (Accuracy up, ROC-AUC collapsed).
  Isolating class weighting as the only changed variable gave a smaller, more believable
  effect.
- Confirmed firsthand that `RandomForestClassifier` results can differ across scikit-learn
  versions (1.8.0 vs 1.9.0) even with identical code, data, and `random_state` — see
  [`requirements.txt`](requirements.txt) for why the version is pinned.
- KNN was the strongest model on both Accuracy and ROC-AUC on the held-out test set — not
  guaranteed in advance, since Module 2's tuning optimized for Accuracy specifically.

## Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter

## How to Run

1. Clone the repo and install dependencies: `pip install -r requirements.txt`
2. Run `module_1_eda_cleaning/module1_eda_cleaning.ipynb` to reproduce the cleaned datasets.
3. Run `module_2_feature_engineering_modeling/module2_feature_engineering_modeling.ipynb` to
   reproduce feature selection, tuning, and the saved models.
4. Run `module_3_evaluation/module3_model_evaluation.ipynb` to reproduce the final evaluation
   and error analysis.
5. Check out [`reports/Diabetes_Prediction_Project_Report.pdf`](https://github.com/Pal0302/diabetes_onset_prediction/blob/main/Reports/Diabetes_Prediction_Project_Report.pdf)
   for the full write-up.
