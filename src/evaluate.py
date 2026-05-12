import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import FIGURES_DIR, REPORTS_DIR, USE_LOG_TARGET


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
