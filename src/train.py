
import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    TimeSeriesSplit,
)

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

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
    "Season",
]

X = df[features]
y = df["PJME_MW"]

if X.isna().any().any() or y.isna().any():
    raise ValueError("Dataset contains missing values. Check preprocessing.")




X_train_all, X_test, y_train_all, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    shuffle=False,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_all,
    y_train_all,
    test_size=0.20,
    shuffle=False,
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))




tscv = TimeSeriesSplit(n_splits=3)




searches = {

    "Linear Regression": LinearRegression(),

    "Ridge Regression": GridSearchCV(
        estimator=Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge()),
        ]),
        param_grid={
            "ridge__alpha": [
                0.01, 0.1, 1, 10, 100, 1000
            ]
        },
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=2,
    ),

    "Random Forest": RandomizedSearchCV(
        estimator=RandomForestRegressor(
            random_state=42,
            n_jobs=1,
        ),
        param_distributions={
            "n_estimators": [50, 100, 150, 200],
            "max_depth": [8, 12, 16, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [0.7, 1.0],
        },
        n_iter=20,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=2,
        verbose=1,
        error_score="raise",
    ),

    "Gradient Boosting": RandomizedSearchCV(
        estimator=HistGradientBoostingRegressor(
            random_state=42,
            early_stopping=False,
        ),
        param_distributions={
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "max_iter": [100, 200, 300, 400],
            "max_leaf_nodes": [15, 31, 63],
            "max_depth": [5, 10, 15, None],
            "min_samples_leaf": [10, 20, 40],
            "l2_regularization": [0.0, 0.1, 1.0],
        },
        n_iter=30,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=2,
        verbose=1,
        error_score="raise",
    ),
}



results = []
trained_models = {}

for name, model in searches.items():

    print(f"\n{'=' * 50}")
    print(f"Training and optimizing: {name}")
    print(f"{'=' * 50}")

    model.fit(X_train, y_train)

    if hasattr(model, "best_estimator_"):
        best_model = model.best_estimator_

        print("\nBest hyperparameters:")
        print(model.best_params_)

        print(
            "Cross-validation MAE:",
            round(-model.best_score_, 2)
        )

    else:
        best_model = model
        print("No hyperparameter tuning needed.")

    y_pred_val = best_model.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred_val)
    mse = mean_squared_error(y_val, y_pred_val)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred_val)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    })

    trained_models[name] = best_model

    print("\nValidation results:")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2:", round(r2, 4))




results_df = pd.DataFrame(results).sort_values(
    by="MAE",
    ascending=True,
).reset_index(drop=True)

print(f"\n{'=' * 50}")
print("MODEL COMPARISON")
print(f"{'=' * 50}")

print(results_df.round(2).to_string(index=False))


best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

print("\nBest model:", best_name)



print("\nRetraining best model on 80% of data...")

best_model.fit(X_train_all, y_train_all)



y_pred_test = best_model.predict(X_test)

test_mae = mean_absolute_error(y_test, y_pred_test)
test_mse = mean_squared_error(y_test, y_pred_test)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_pred_test)

print(f"\n{'=' * 50}")
print("FINAL TEST EVALUATION")
print(f"{'=' * 50}")

print("Best model:", best_name)
print("MAE:", round(test_mae, 2))
print("MSE:", round(test_mse, 2))
print("RMSE:", round(test_rmse, 2))
print("R2:", round(test_r2, 4))



models_dir = project_dir / "models"
reports_dir = project_dir / "reports"

models_dir.mkdir(parents=True, exist_ok=True)
reports_dir.mkdir(parents=True, exist_ok=True)

model_path = models_dir / "best_forecasting_model.joblib"

joblib.dump(best_model, model_path)

results_df.to_csv(
    reports_dir / "model_comparison.csv",
    index=False,
)

test_predictions = pd.DataFrame({
    "Datetime": df.loc[X_test.index, "Datetime"].values,
    "Actual_MW": y_test.values,
    "Predicted_MW": y_pred_test,
})

test_predictions.to_csv(
    reports_dir / "test_predictions.csv",
    index=False,
)

metadata = {
    "best_model": best_name,
    "features": features,
    "test_mae": float(test_mae),
    "test_mse": float(test_mse),
    "test_rmse": float(test_rmse),
    "test_r2": float(test_r2),
}

with open(
    reports_dir / "model_metadata.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(metadata, f, indent=4)

print("\nTraining completed!")
print("Model saved:", model_path)
print("Comparison saved:", reports_dir / "model_comparison.csv")
print("Predictions saved:", reports_dir / "test_predictions.csv")
