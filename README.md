# CPSC483 Final Project - Real Estate Price Predictor

A machine learning project that predicts house sale prices using the Ames Housing dataset.
The project uses **regression** to predict an exact sales price and **classification**
predict a price tier: Low Tier / Medium Tier / High Tier / Luxury.

---

## Team
* Sebastian Pfander
* Vincent Polanco

---

## Project Overview

### Tasks
| Task | Description | Models |
|------|-------------|--------|
| Regression | Predict the exact sale price in dollars | Ridge Regression vs. Random Forest Regressor |
| Classification | Predict the price tier (4 categories) | Logistic Regression vs. Random Forest Classifier |

### Price Tiers (Classification)
| Tier | Label | Price Range |
|------|-------|-------------|
| 0 | Low Tier | < $130,000 |
| 1 | Medium Tier | $130,000 - $163,000 |
| 2 | High Tier | $163,000 - $215,000 |
| 3 | Luxury | > $215,000 |

Boundaries are based on the quartiles of the Ames Housing dataset, for producing roughly balanced classes.

---

## Dataset

**Ames Housing Dataset** from Kaggle:
- Competition: `house-prices-advanced-regression-techniques`
- ~1,460 training samples with 79 features describing residential homes in Ames, Iowa
- Target column: `SalePrice`

### Download Instructions
1. Go to the Kaggle competition page and download the dataset
   - https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview
2. Place the following files in `data/kaggle_data/`:
   - `train.csv` — required for training and evaluation
   - `data_description.txt` — feature dictionary (recommended reference for the paper)

The `data/` directory is git-ignored — do not commit these files.

## Project Structure

```
CPSC483_final_project/
├── data/kaggle_data/           # Dataset (git-ignored)
│   └── train.csv
├── models/                     # Saved models (git-ignored)
├── outputs/
│   ├── figures/                # Plots (git-ignored)
│   └── reports/                # Metrics CSVs (git-ignored)
├── src/
│   ├── config.py               # Paths and constants
│   ├── preprocessing.py        # Data loading and feature pipeline
│   ├── train_regression.py     # Ridge and Random Forest Regressor
│   ├── train_classification.py # Logistic Regression and Random Forest Classifier
│   └── evaluate.py             # Metrics, plots, and reports
└── run_pipeline.py             # Main entry point
```

---

## Setup

### Requirements
- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Pipeline

From the project root directory, run:

```bash
python .\run_pipeline.py
```

This will:
1. Load and preprocess the Ames Housing data
2. Train and tune Ridge Regression and Random Forest Regressor (regression)
3. Train and tune Logistic Regression and Random Forest Classifier (classification)
4. Generate all evaluation metrics, confusion matrices, and ROC curves
5. Save trained models to `models/` and plots to `outputs/figures/`

---

## Outputs

After running the pipeline, three directories are created (all git-ignored):

**`models/`** — trained sklearn pipelines (preprocessor + model), saved with joblib
- `ridge.joblib`
- `random_forest_reg.joblib`
- `logistic_regression.joblib`
- `random_forest_clf.joblib`

**`outputs/figures/`** — evaluation plots
- `reg_pred_vs_actual_ridge.png`
- `reg_pred_vs_actual_random_forest_reg.png`
- `clf_confusion_logistic_regression.png`
- `clf_confusion_random_forest_clf.png`
- `clf_roc_logistic_regression.png`
- `clf_roc_random_forest_clf.png`

**`outputs/reports/`** — evaluation metrics
- `regression_metrics.csv` — RMSE, MAE, and R² for both regression models
- `classification_metrics.csv` — Accuracy, Precision, Recall, F1, and ROC AUC for both classifiers