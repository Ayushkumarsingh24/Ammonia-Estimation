"""Compare single-model baselines with 5-fold cross-validation.

The StandardScaler lives INSIDE each Pipeline, so it is re-fit on each fold's
training data only (no data leakage).
"""
import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from config import CLEAN_DATA, FEATURES, TARGET, RESULTS_DIR, RANDOM_STATE

MODELS = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE),
    "SVR (RBF)": SVR(kernel="rbf", C=100, epsilon=0.1),
    "LightGBM (single stage)": lgb.LGBMRegressor(
        objective="regression", n_estimators=50, learning_rate=0.1,
        num_leaves=16, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1),
}

SCORING = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2",
}


def main():
    df = pd.read_csv(CLEAN_DATA)
    X, y = df[FEATURES], df[TARGET]
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for name, model in MODELS.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        s = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
        rows.append({
            "Model": name,
            "MAE": -s["test_MAE"].mean(),
            "RMSE": -s["test_RMSE"].mean(),
            "R2": s["test_R2"].mean(),
            "R2 std": s["test_R2"].std(),
        })
        print(f"{name:<26} R2={rows[-1]['R2']:.3f}  MAE={rows[-1]['MAE']:.3f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = pd.DataFrame(rows).round(4)
    out.to_csv(RESULTS_DIR / "baseline_cv_results.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
