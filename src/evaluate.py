import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from src.config import FIGURES_DIR, REPORTS_DIR, TIER_LABELS, USE_LOG_TARGET


def _ensure_dirs():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _inverse_target(values):
    return np.expm1(values) if USE_LOG_TARGET else np.asarray(values)


def evaluate_regression(models, X_test, y_test):
    """
    Score each regression model on the test set, save a metrics CSV,
    and write a predicted-vs-actual scatter plot per model.
    """
    _ensure_dirs()
    actual_prices = _inverse_target(y_test)

    metric_rows = []
    for model_name, model in models.items():
        predicted_prices = _inverse_target(model.predict(X_test))
        rmse = float(np.sqrt(mean_squared_error(actual_prices, predicted_prices)))
        mae = float(mean_absolute_error(actual_prices, predicted_prices))
        r2 = float(r2_score(actual_prices, predicted_prices))
        metric_rows.append({"model": model_name, "rmse": rmse, "mae": mae, "r2": r2})

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(actual_prices, predicted_prices, alpha=0.4, s=12)
        axis_min = float(min(actual_prices.min(), predicted_prices.min()))
        axis_max = float(max(actual_prices.max(), predicted_prices.max()))
        ax.plot([axis_min, axis_max], [axis_min, axis_max], "r--", linewidth=1)
        ax.set_xlabel("Actual SalePrice")
        ax.set_ylabel("Predicted SalePrice")
        ax.set_title(f"{model_name}: predicted vs. actual")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"reg_pred_vs_actual_{model_name}.png", dpi=150)
        plt.close(fig)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(REPORTS_DIR / "regression_metrics.csv", index=False)
    return metrics_df


def evaluate_classification(models, X_test, y_test):
    _ensure_dirs()
    classes = [0, 1, 2, 3]
    y_test_bin = label_binarize(y_test, classes=classes)

    metric_rows = []
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        metric_rows.append({
            "model": model_name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_weighted": float(precision_score(y_test, y_pred, average="weighted")),
            "recall_weighted": float(recall_score(y_test, y_pred, average="weighted")),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
            "roc_auc_ovr": float(roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="weighted")),
        })

        # Confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, y_pred),
            display_labels=TIER_LABELS,
        ).plot(ax=ax, colorbar=False)
        ax.set_title(f"{model_name}: confusion matrix")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"clf_confusion_{model_name}.png", dpi=150)
        plt.close(fig)

        # ROC curves
        fig, ax = plt.subplots(figsize=(7, 5))
        for i, label in enumerate(TIER_LABELS):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            auc = roc_auc_score(y_test_bin[:, i], y_prob[:, i])
            ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.2f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{model_name}: ROC curves (one-vs-rest)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"clf_roc_{model_name}.png", dpi=150)
        plt.close(fig)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(REPORTS_DIR / "classification_metrics.csv", index=False)
    return metrics_df
