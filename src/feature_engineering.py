
import pandas as pd
from pathlib import Path


def create_features(df):
    df_features = df.copy()

    df_features["Datetime"] = pd.to_datetime(df_features["Datetime"])

    df_features = df_features.sort_values("Datetime").reset_index(drop=True)

    # Extract the hour (0-23)
    df_features["Hour"] = df_features["Datetime"].dt.hour

    # Extract day of week (0 = Monday, 6 = Sunday)
    df_features["DayOfWeek"] = df_features["Datetime"].dt.dayofweek

    # Extract month (1-12)
    df_features["Month"] = df_features["Datetime"].dt.month

    # Extract quarter (1-4)
    df_features["Quarter"] = df_features["Datetime"].dt.quarter

    # Extract year
    df_features["Year"] = df_features["Datetime"].dt.year

    # Extract day of year (1-366)
    df_features["DayOfYear"] = df_features["Datetime"].dt.dayofyear

    # 1 if Saturday or Sunday, otherwise 0
    df_features["IsWeekend"] = (
        df_features["DayOfWeek"] >= 5
    ).astype(int)

    # 1 if hour is between 09:00 and 17:00
    df_features["IsBusinessHour"] = (
        df_features["Hour"].between(9, 17)
    ).astype(int)

    # 0 = Winter, 1 = Spring, 2 = Summer, 3 = Autumn
    df_features["Season"] = (
        df_features["Month"] % 12 // 3
    )

    return df_features


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[1]

    input_path = project_dir / "data/processed/PJME_cleaned.csv"
    output_path = project_dir / "data/processed/PJME_features.csv"

    df = pd.read_csv(input_path, parse_dates=["Datetime"])

    df_features = create_features(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path, index=False)

    print("Feature engineering completed!")
    print("\nFirst 5 rows:")
    print(df_features.head())

    print("\nDataset shape:", df_features.shape)

    print("\nMissing values:")
    print(df_features.isna().sum())

    print("\nSaved to:", output_path)
