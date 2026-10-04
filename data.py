"""Load the cleaned data and build the shared train/test split."""
import pandas as pd
from sklearn.model_selection import train_test_split

from config import CLEAN_DATA, FEATURES, TARGET, RANDOM_STATE, TEST_SIZE


def load_split():
    df = pd.read_csv(CLEAN_DATA)

    # Stage 1 target: is ammonia detectable (> 0)?
    y_class = (df[TARGET] > 0).astype(int)
    X = df[FEATURES]
    y_reg = df[TARGET]

    X_train, X_test, yc_train, yc_test = train_test_split(
        X, y_class,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_class,
    )
    yr_train = y_reg.loc[X_train.index]
    yr_test = y_reg.loc[X_test.index]
    return X_train, X_test, yc_train, yc_test, yr_train, yr_test
