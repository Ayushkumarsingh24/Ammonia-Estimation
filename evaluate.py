"""Evaluate the two-stage model on the held-out test set and save plots + metrics."""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
)

from config import MODELS_DIR, RESULTS_DIR
from data import load_split
from predict import two_stage_predict


def regression_metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    _, X_test, _, yc_test, _, yr_test = load_split()
    stage1 = joblib.load(MODELS_DIR / "stage1_model.pkl")
    stage2 = joblib.load(MODELS_DIR / "stage2_model.pkl")

    final_pred, class_pred, _ = two_stage_predict(stage1, stage2, X_test)

    # ---- Stage 1 ----
    acc = accuracy_score(yc_test, class_pred)
    print("Stage 1 accuracy:", round(acc, 4))
    print(classification_report(
        yc_test, class_pred, target_names=["Not Detectable", "Detectable"]))

    cm = confusion_matrix(yc_test, class_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
                xticklabels=["Not Detectable", "Detectable"],
                yticklabels=["Not Detectable", "Detectable"])
    plt.xlabel("Predicted Label"); plt.ylabel("True Label")
    plt.title("Stage 1 Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "stage1_confusion_matrix.png", dpi=150)
    plt.close()

    # ---- Stage 2 alone, on truly detectable test rows ----
    pos = yc_test == 1
    stage2_only = regression_metrics(yr_test[pos], stage2.predict(X_test[pos]))

    # ---- End-to-end two-stage ----
    two_stage = regression_metrics(yr_test, final_pred)

    # ---- Oracle: use the TRUE detectability label (diagnostic only) ----
    oracle_pred = np.zeros(len(X_test))
    idx = np.where(yc_test.values == 1)[0]
    oracle_pred[idx] = stage2.predict(X_test.iloc[idx])
    oracle = regression_metrics(yr_test, np.maximum(oracle_pred, 0))

    metrics = {
        "stage1_accuracy": float(acc),
        "stage2_on_detectable_rows": stage2_only,
        "two_stage_end_to_end": two_stage,
        "oracle_two_stage": oracle,
    }
    print(json.dumps(metrics, indent=2))
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- Actual vs predicted ----
    plt.figure(figsize=(8, 6))
    plt.scatter(yr_test, final_pred, alpha=0.6)
    m = max(yr_test.max(), final_pred.max())
    plt.plot([0, m], [0, m], "--", label="Ideal 1:1 Line")
    plt.xlabel("Actual Ammonia Concentration")
    plt.ylabel("Predicted Ammonia Concentration")
    plt.title("Actual vs Predicted Ammonia Concentration")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
