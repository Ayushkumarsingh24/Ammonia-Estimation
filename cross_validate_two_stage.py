"""5-fold stratified cross-validation of the full two-stage pipeline.

This gives a like-for-like comparison with baselines.py (same CV protocol),
instead of relying on a single 80/20 split.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import StratifiedKFold

from config import (
    CLEAN_DATA, FEATURES, TARGET, RESULTS_DIR, RANDOM_STATE,
    STAGE1_PARAMS, STAGE2_PARAMS,
)
from predict import two_stage_predict


def main():
    df = pd.read_csv(CLEAN_DATA)
    X, y = df[FEATURES], df[TARGET]
    y_class = (y > 0).astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = []

    for k, (tr, te) in enumerate(skf.split(X, y_class), start=1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        yc_tr, yc_te = y_class.iloc[tr], y_class.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        stage1 = lgb.LGBMClassifier(**STAGE1_PARAMS).fit(X_tr, yc_tr)
        mask = yc_tr == 1
        stage2 = lgb.LGBMRegressor(**STAGE2_PARAMS).fit(X_tr[mask], y_tr[mask])

        pred, class_pred, _ = two_stage_predict(stage1, stage2, X_te)
        folds.append({
            "fold": k,
            "detect_acc": accuracy_score(yc_te, class_pred),
            "R2": r2_score(y_te, pred),
            "MAE": mean_absolute_error(y_te, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
        })
        print(folds[-1])

    res = pd.DataFrame(folds)
    summary = {c: {"mean": float(res[c].mean()), "std": float(res[c].std())}
               for c in ["detect_acc", "R2", "MAE", "RMSE"]}
    print(json.dumps(summary, indent=2))

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "two_stage_cv_results.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
