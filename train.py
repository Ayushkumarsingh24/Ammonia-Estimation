"""Train the two-stage LightGBM model and save both stages."""
import joblib
import lightgbm as lgb

from config import MODELS_DIR, STAGE1_PARAMS, STAGE2_PARAMS
from data import load_split


def main():
    X_train, X_test, yc_train, yc_test, yr_train, yr_test = load_split()
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # Stage 1: detectable vs not detectable
    stage1 = lgb.LGBMClassifier(**STAGE1_PARAMS)
    stage1.fit(X_train, yc_train)

    # Stage 2: concentration, trained only on rows where ammonia is detectable
    mask = yc_train == 1
    stage2 = lgb.LGBMRegressor(**STAGE2_PARAMS)
    stage2.fit(X_train[mask], yr_train[mask])
    print(f"Stage 2 trained on {int(mask.sum())} detectable rows")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(stage1, MODELS_DIR / "stage1_model.pkl")
    joblib.dump(stage2, MODELS_DIR / "stage2_model.pkl")
    print("Saved models to", MODELS_DIR)


if __name__ == "__main__":
    main()
