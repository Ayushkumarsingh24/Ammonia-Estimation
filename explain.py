"""Feature importance (gain, permutation) and SHAP plots for both stages."""
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from config import MODELS_DIR, RESULTS_DIR
from data import load_split


def save(name):
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    _, X_test, _, yc_test, _, yr_test = load_split()
    stage1 = joblib.load(MODELS_DIR / "stage1_model.pkl")
    stage2 = joblib.load(MODELS_DIR / "stage2_model.pkl")

    pos = yc_test == 1
    X_pos, y_pos = X_test[pos], yr_test[pos]

    # ---- Stage 2: gain importance ----
    gain = pd.Series(
        stage2.booster_.feature_importance(importance_type="gain"),
        index=X_test.columns,
    ).sort_values()
    plt.figure(figsize=(7, 5))
    plt.barh(gain.index, gain.values)
    plt.xlabel("Total Gain"); plt.title("LightGBM Gain-Based Feature Importance")
    save("stage2_gain_importance")

    # ---- Stage 2: permutation importance ----
    perm = permutation_importance(
        stage2, X_pos, y_pos, n_repeats=10, random_state=42,
        scoring="neg_mean_absolute_error",
    )
    perm_df = pd.DataFrame({
        "Feature": X_pos.columns,
        "Importance": perm.importances_mean,
        "Std": perm.importances_std,
    }).sort_values("Importance")
    plt.figure(figsize=(7, 5))
    plt.barh(perm_df["Feature"], perm_df["Importance"], xerr=perm_df["Std"])
    plt.xlabel("Decrease in Model Performance")
    plt.title("Permutation Feature Importance - Stage 2")
    save("stage2_permutation_importance")

    # ---- SHAP: Stage 1 (classifier) ----
    sv1 = shap.TreeExplainer(stage1).shap_values(X_test)
    if isinstance(sv1, list):      # older SHAP returns one array per class
        sv1 = sv1[1]
    for kind in ("bar", "dot"):
        shap.summary_plot(sv1, X_test, plot_type=kind, show=False)
        save(f"stage1_shap_{kind}")

    # ---- SHAP: Stage 2 (regressor) ----
    sv2 = shap.TreeExplainer(stage2).shap_values(X_pos)
    for kind in ("bar", "dot"):
        shap.summary_plot(sv2, X_pos, plot_type=kind, show=False)
        save(f"stage2_shap_{kind}")

    print("Saved plots to", RESULTS_DIR)


if __name__ == "__main__":
    main()
