import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

project_dir = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    project_dir / "data/processed/PJME_features.csv",
    parse_dates=["Datetime"]
)

df = df.sort_values("Datetime").reset_index(drop=True)

features = [
    "Hour",
    "DayOfWeek",
    "Month",
    "Quarter",
    "Year",
    "DayOfYear",
    "IsWeekend",
    "IsBusinessHour",
    "Season"
]

X = df[features]
y = df["PJME_MW"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)