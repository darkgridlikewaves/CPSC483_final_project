# CPSC483 Final Project - Real Estate Price Predictor

A machine learning project that predicts house sale prices using the Ames Housing dataset.
The project uses **regression** to predict an exact sales price and **classification**
to predict a price tier: Low Tier / Medium Tier / High Tier / Luxury.

---

## Team
* Sebastian Pfander
* Vincent Polanco
* Ethan Santos

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
├── notebooks/                  # Exploratory Data Analysis
├── outputs/
│   ├── figures/                # Plots (git-ignored)
│   └── reports/                # Metrics CSVs (git-ignored)
├── src/
│   ├── config.py               # Paths and constants
│   ├── preprocessing.py        # Data loading and feature pipeline
│   ├── train_regression.py     # Ridge and Random Forest Regressor
│   ├── train_classification.py # Logistic Regression and Random Forest Classifier
│   ├── evaluate.py             # Metrics, plots, and reports
│   ├── tune_models.py          # Hyperparameter testing
│   └── utils.py                # Shared helpers (tier labeling, console output)
├── run_pipeline.py             # Main entry point
├── predict.py                  # CLI tool for single-house predictions
└── app.py                      # Streamlit interactive web app
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


### Making a Single Prediction (CLI)

After training the models, use `predict.py` to get a price prediction for a specific house:

```bash
# Predict price and tier using key features
python predict.py --overall-quality 8 --grade-living-area 2100 --garage-cars 2 --year-built 2003

# Regression only with verbose output
python predict.py --overall-quality 7 --grade-living-area 1800 --total-basement 900 --task regression --verbose

# Classification only using Logistic Regression
python predict.py --overall-quality 5 --grade-living-area 1200 --task classification --model logistic
```

| Argument | Type | Description |
|----------|------|-------------|
| `--overall-quality` | int (1-10) | Overall material and finish quality |
| `--grade-living-area` | int | Above-grade living area (sq ft) |
| `--garage-cars` | int | Garage capacity (number of cars) |
| `--total-basement` | int | Total basement area (sq ft) |
| `--year-built` | int | Original construction year |
| `--neighborhood` | str | Neighborhood name (e.g. NridgHt, OldTown) |
| `--bedrooms` | int | Bedrooms above basement level |
| `--full-bath` | int | Number of full bathrooms |
| `--task` | str | `regression`, `classification`, or `both` (default: both) |
| `--model` | str | `ridge`, `rf`, or `logistic` (default: rf) |
| `--verbose` | flag | Print confidence scores and extra detail |

---

### Running the Web App (Demo)

Once you have trained the models by running the pipeline, you can launch the interactive web interface to test the predictions.
From the project root directory, run:

```bash
streamlit run app.py
```

*(Note for Windows users: If the command above is not recognized, run `python -m streamlit run app.py` instead).*

This will open a local web browser page where you can input specific housing features (like square footage and neighborhood) and receive a real-time price and tier prediction from the trained Random Forest models.

### Viewing the Exploratory Data Analysis (EDA)

The data exploration and visualization phase is documented in a Jupyter Notebook. 

To view the analysis, charts, and correlation matrix:
1. Navigate to the `notebooks/` directory in this repository.
2. Click on `01_eda.ipynb` to view it directly in your browser via GitHub.
3. Alternatively, you can open the file locally using VS Code (with the Jupyter extension installed) or JupyterLab.

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

---

## Results

### Regression (test set, 20% holdout)
| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| Ridge Regression | $30,104 | $19,009 | 0.882 |
| Random Forest Regressor | $30,461 | $17,801 | 0.879 |

### Classification (test set, 20% holdout)
| Model | Accuracy | F1 (weighted) | ROC-AUC |
|-------|----------|---------------|---------|
| Logistic Regression | 78.1% | 0.779 | 0.945 |
| Random Forest Classifier | 76.7% | 0.766 | 0.944 |

All models tuned with GridSearchCV and 5-fold cross-validation.
