"""Clean the raw Pond12 sensor data and save it to data/processed/."""
import pandas as pd

from config import RAW_DATA, CLEAN_DATA

# Columns with no predictive value (identifiers, timestamps, or constant values)
DROP_COLS = ["created_at", "entry_id", "Population", "Weight", "Length"]


def iqr_outlier_index(df, column):
    """Return the index of rows outside Q1 - 1.5*IQR / Q3 + 1.5*IQR."""
    q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return df[(df[column] < low) | (df[column] > high)].index


def clean(df):
    df = df.drop(columns=DROP_COLS, errors="ignore")

    # DS18B20 returns -127 when the sensor is disconnected -> not a real reading
    df = df[df["TEMPERATURE"] != -127]

    # Remove extreme AMMONIA spikes (IQR rule)
    df = df.drop(index=iqr_outlier_index(df, "AMMONIA"))

    # A handful of NITRATE readings were far outside the normal range
    df = df[df["NITRATE"] <= 3000]

    return df


def main():
    df = pd.read_csv(RAW_DATA)
    print("Raw shape  :", df.shape)

    df = clean(df)
    print("Clean shape:", df.shape)

    CLEAN_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DATA, index=False)
    print("Saved ->", CLEAN_DATA)


if __name__ == "__main__":
    main()
