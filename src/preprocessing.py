import pandas as pd
from pathlib import Path


def preprocess(df):
    df_clean = df[["Datetime", "PJME_MW"]].copy()

    df_clean["Datetime"] = pd.to_datetime(
        df_clean["Datetime"],
        errors="coerce"
    )

    df_clean["PJME_MW"] = pd.to_numeric(
        df_clean["PJME_MW"],
        errors="coerce"
    )

    Q1 = df_clean["PJME_MW"].quantile(0.25)
    Q3 = df_clean["PJME_MW"].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df_clean[
        (df_clean["PJME_MW"] < lower) |
        (df_clean["PJME_MW"] > upper)
    ]

    print("Lower bound:", lower)
    print("Upper bound:", upper)
    print("Potential outliers:", len(outliers))

    df_clean = df_clean.drop_duplicates()

    df_clean = df_clean.sort_values(
        "Datetime"
    ).reset_index(drop=True)

    print("\nOriginal shape:", df.shape)
    print("Cleaned shape:", df_clean.shape)

    print("\nData types:")
    print(df_clean.dtypes)

    print("\nMissing values:")
    print(df_clean.isna().sum())

    print("\nDuplicate timestamps:")
    print(df_clean["Datetime"].duplicated().sum())

    return df_clean


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[1]

    input_path = project_dir / "data/raw/pjme-hourly.csv"
    output_path = project_dir / "data/processed/PJME_cleaned.csv"

    df = pd.read_csv(input_path)

    df_clean = preprocess(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)

    print("\nDataset saved!")