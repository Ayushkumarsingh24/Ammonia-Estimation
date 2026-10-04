"""Shared settings for the ammonia estimation pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DATA = ROOT / "data" / "raw" / "pond12.csv"
CLEAN_DATA = ROOT / "data" / "processed" / "cleaned_data.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIRMWARE_DIR = ROOT / "firmware"

# Model inputs. ORDER MATTERS: the ESP32 firmware uses this exact order.
FEATURES = ["TEMPERATURE", "TURBIDITY", "DISOLVED OXYGEN", "pH", "NITRATE"]
TARGET = "AMMONIA"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Stage 1: detectable / not detectable classifier
STAGE1_PARAMS = dict(
    n_estimators=50,
    num_leaves=13,
    max_depth=5,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    verbosity=-1,
)

# Stage 2: ammonia concentration regressor (trained on detectable rows only)
STAGE2_PARAMS = dict(
    n_estimators=55,
    learning_rate=0.25,
    num_leaves=32,
    max_depth=8,
    min_child_samples=5,
    subsample=0.80,
    colsample_bytree=1.0,
    reg_lambda=5,
    random_state=RANDOM_STATE,
    verbosity=-1,
)
