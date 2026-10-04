"""Two-stage prediction helper, shared by evaluate.py and for inference."""
import numpy as np


def two_stage_predict(stage1, stage2, X):
    """Stage 1 decides detectable/not; Stage 2 predicts concentration if detectable."""
    detectable = stage1.predict(X)
    prob = stage1.predict_proba(X)[:, 1]

    pred = np.zeros(len(X))
    idx = np.where(detectable == 1)[0]
    if len(idx):
        pred[idx] = stage2.predict(X.iloc[idx])

    return np.maximum(pred, 0), detectable, prob
