import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 50)

RANDOM_STATE = 42
OUTPUT_DIR = Path("rainfall_project_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("All libraries imported successfully!")
print("Output folder:", OUTPUT_DIR.resolve())



DATA_PATH = Path("/Users/deepanshuyadav/Desktop/Predictive_analysis/Rainfall_Project/rainfall_project_outputs/rainfall_dataset.csv")

# Also check the common uploaded-file location.
if not DATA_PATH.exists():
    DATA_PATH = Path("/Users/deepanshuyadav/Desktop/Predictive_analysis/Rainfall_Project/rainfall_project_outputs/rainfall_dataset.csv")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Dataset not found. Put 'rainfall dataset.csv' "
        "in the same folder as your notebook or update DATA_PATH."
    )

df = pd.read_csv("/Users/deepanshuyadav/Desktop/Predictive_analysis/Rainfall_Project/rainfall_project_outputs/rainfall_dataset.csv")

# Standardize column names.
df.columns = df.columns.str.strip()

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print(df.head())

print("Dataset dimensions:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nFirst five rows:")
print(df.head())

print("\nLast five rows:")
print(df.tail())

print("\nStatistical summary:")
print(df.describe(include="all").T)

print("\nMissing values in each column:")
print(
    df.isnull().sum()
    .sort_values(ascending=False)
    .to_frame("Missing Values")
)

print("\nExact duplicate rows:", df.duplicated().sum())




# Work on a copy of the original dataset.
df = df.copy()

# Remove extra spaces from text fields.
df["DIVISION"] = df["DIVISION"].astype("string").str.strip()

# Convert year to numeric.
df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")

# Columns expected in this dataset.
monthly_cols = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
]

seasonal_cols = ["Jan-Feb", "Mar-May", "Jun-Sep", "Oct-Dec"]

rainfall_cols = monthly_cols + seasonal_cols + ["ANNUAL"]

# Keep only rainfall columns that exist in the CSV.
rainfall_cols = [c for c in rainfall_cols if c in df.columns]

# Convert rainfall columns to numeric.
for col in rainfall_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove exact duplicate records.
before = len(df)
df = df.drop_duplicates()
print("Duplicate rows removed:", before - len(df))

# Remove rows with invalid year or missing division.
df = df.dropna(subset=["DIVISION", "YEAR"])
df["YEAR"] = df["YEAR"].astype(int)

# Rainfall cannot be negative. Treat negative values as invalid.
for col in rainfall_cols:
    df.loc[df[col] < 0, col] = np.nan

# Keep one record per division per year.
duplicate_keys = df.duplicated(subset=["DIVISION", "YEAR"], keep=False)
print("Rows involved in duplicate division-year records:", duplicate_keys.sum())

# Resolve duplicate division-year records by averaging numeric observations.
if duplicate_keys.any():
    numeric_cols = [c for c in rainfall_cols if c in df.columns]
    df = df.groupby(
        ["DIVISION", "YEAR"], as_index=False
    )[numeric_cols].mean().merge(
        df[["DIVISION", "YEAR"]].drop_duplicates(),
        on=["DIVISION", "YEAR"],
        how="right"
    )

# Sort chronologically before creating lag features.
df = df.sort_values(["DIVISION", "YEAR"]).reset_index(drop=True)

print("\nCleaned dataset shape:", df.shape)
print("\nMissing values after basic cleaning:")
print(df.isnull().sum().to_frame("Missing Values"))

print("\nCleaned data preview:")
print(df.head())



#EDA
##Handling the missing values
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing) > 0:
    plt.figure(figsize=(12, 5))
    missing.plot(kind="bar")
    plt.title("Missing Values by Column")
    plt.xlabel("Columns")
    plt.ylabel("Number of Missing Values")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "missing_values.png", dpi=300)
    plt.show()
else:
    print("No missing values remain.")



##Avg annual rainfall trend
annual_trend = df.groupby("YEAR")["ANNUAL"].mean()

plt.figure(figsize=(12, 5))

plt.plot(
    annual_trend.index,
    annual_trend.values,
    linewidth=1.5
)

plt.title("Historical Average Annual Rainfall")
plt.xlabel("Year")
plt.ylabel("Average Annual Rainfall (mm)")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "annual_rainfall_trend.png", dpi=300)
plt.show()



##Distribution of annual rainfall
plt.figure(figsize=(10, 5))
sns.histplot(df["ANNUAL"].dropna(), bins=40, kde=True)
plt.title("Distribution of Annual Rainfall")
plt.xlabel("Annual Rainfall (mm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "annual_rainfall_distribution.png", dpi=300)
plt.show()


##Avg rainfall by month
available_months = [c for c in monthly_cols if c in df.columns]

monthly_average = df[available_months].mean()

plt.figure(figsize=(12, 5))
sns.barplot(x=monthly_average.index, y=monthly_average.values)
plt.title("Average Rainfall by Month")
plt.xlabel("Month")
plt.ylabel("Average Rainfall (mm)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "monthly_rainfall.png", dpi=300)
plt.show()

print(monthly_average.to_frame("Average Rainfall (mm)"))



##Divisions with the highest average annual rainfall
division_rainfall = (
    df.groupby("DIVISION")["ANNUAL"]
    .mean()
    .dropna()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 6))
division_rainfall.sort_values().plot(kind="barh")
plt.title("Top 10 Divisions by Average Annual Rainfall")
plt.xlabel("Average Annual Rainfall (mm)")
plt.ylabel("Division")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_rainfall_divisions.png", dpi=300)
plt.show()

print(division_rainfall.to_frame("Average Annual Rainfall (mm)"))

#Correlation between numeric rainfall variables
numeric_df = df.select_dtypes(include=np.number)
correlation = numeric_df.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0,
    annot=False
)
plt.title("Correlation Heatmap of Numeric Variables")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=300)
plt.show()




#Feature engineering
model_df = df.copy()

## Sort records before shifting.
model_df = model_df.sort_values(
    ["DIVISION", "YEAR"]
).reset_index(drop=True)

grouped = model_df.groupby("DIVISION", sort=False)

# Annual rainfall from previous years.
for lag in [1, 2, 3]:
    model_df[f"ANNUAL_LAG_{lag}"] = (
        grouped["ANNUAL"].shift(lag)
    )

# Rolling statistics based only on previous years.
model_df["ROLLING_MEAN_3Y"] = (
    grouped["ANNUAL"]
    .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
)

model_df["ROLLING_STD_3Y"] = (
    grouped["ANNUAL"]
    .transform(lambda s: s.shift(1).rolling(3, min_periods=2).std())
)

# Monthly rainfall from the previous year.
for month in available_months:
    model_df[f"PREVIOUS_YEAR_{month}"] = (
        model_df.groupby("DIVISION", sort=False)[month].shift(1)
    )

# Require at least one previous annual rainfall observation.
model_df = model_df.dropna(subset=["ANNUAL_LAG_1"])

# A target value is required to train and evaluate a model.
model_df = model_df.dropna(subset=["ANNUAL"])

model_df = model_df.replace([np.inf, -np.inf], np.nan)

print("Shape after feature engineering:", model_df.shape)
print("\nNew features:")
print([
    c for c in model_df.columns
    if "LAG" in c or "ROLLING" in c or "PREVIOUS_YEAR" in c
])

print(model_df.head())





#Split the dataset into training and testing sets

feature_cols = (
    ["DIVISION", "YEAR",
     "ANNUAL_LAG_1", "ANNUAL_LAG_2", "ANNUAL_LAG_3",
     "ROLLING_MEAN_3Y", "ROLLING_STD_3Y"]
    + [f"PREVIOUS_YEAR_{m}" for m in available_months]
)

target_col = "ANNUAL"

# Chronological split.
train_df = model_df[model_df["YEAR"] <= 2000].copy()
test_df = model_df[model_df["YEAR"] > 2000].copy()

X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Training years:", train_df["YEAR"].min(), "-", train_df["YEAR"].max())
print("Testing years:", test_df["YEAR"].min(), "-", test_df["YEAR"].max())

if X_train.empty or X_test.empty:
    raise ValueError(
        "Training or testing set is empty. Check the years in your dataset."
    )

print("\nTraining and testing sets created successfully.")




#Train five machine-learning models

numeric_features = [
    c for c in feature_cols if c != "DIVISION"
]
categorical_features = ["DIVISION"]


def make_one_hot_encoder():
    # Supports both newer and older scikit-learn versions.
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


def make_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", make_one_hot_encoder())
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])


models = {
    "Ridge Regression": Ridge(alpha=10.0),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE
    ),

    "SVR": TransformedTargetRegressor(
        regressor=Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", SVR(kernel="rbf", C=10, epsilon=0.1))
        ]),
        transformer=StandardScaler()
    )
}

trained_models = {}
predictions = {}
results = []

for name, estimator in models.items():
    print(f"\nTraining {name}...")

    if name == "SVR":
        model = clone(estimator)
    else:
        model = Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", clone(estimator))
        ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    trained_models[name] = model
    predictions[name] = y_pred

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.4f}")

print("\nAll five models trained successfully.")




#Evaluate and compare the models
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("RMSE").reset_index(drop=True)

print("Model comparison:")
print(results_df.round(4))

results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

best_model_name = results_df.loc[0, "Model"]
best_model = trained_models[best_model_name]
best_predictions = predictions[best_model_name]

print("\nBest model by RMSE:", best_model_name)

#Compare model performance visually
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.barplot(data=results_df, x="Model", y="MAE", ax=axes[0])
axes[0].set_title("MAE Comparison")
axes[0].tick_params(axis="x", rotation=65)

sns.barplot(data=results_df, x="Model", y="RMSE", ax=axes[1])
axes[1].set_title("RMSE Comparison")
axes[1].tick_params(axis="x", rotation=65)

sns.barplot(data=results_df, x="Model", y="R2", ax=axes[2])
axes[2].set_title("R² Comparison")
axes[2].tick_params(axis="x", rotation=65)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=300)
plt.show()


#Visualize predictions and feature importance

##Actual vs predicted rainfall
plt.figure(figsize=(8, 7))

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.5
)

min_value = min(y_test.min(), np.min(best_predictions))
max_value = max(y_test.max(), np.max(best_predictions))

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Annual Rainfall (mm)")
plt.ylabel("Predicted Annual Rainfall (mm)")
plt.title(f"Actual vs Predicted Rainfall — {best_model_name}")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=300)
plt.show()


# Generate prediction graphs for the remaining four models

remaining_models = [
    "Ridge Regression",
    "XGBoost",
    "HistGradientBoosting",
    "SVR"
]

for model_name in remaining_models:

    model_predictions = predictions[model_name]

    # 1. Actual vs Predicted Scatter Plot
    plt.figure(figsize=(8, 7))

    plt.scatter(
        y_test,
        model_predictions,
        alpha=0.5
    )

    min_value = min(y_test.min(), np.min(model_predictions))
    max_value = max(y_test.max(), np.max(model_predictions))

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    plt.xlabel("Actual Annual Rainfall (mm)")
    plt.ylabel("Predicted Annual Rainfall (mm)")
    plt.title(f"Actual vs Predicted Rainfall — {model_name}")
    plt.tight_layout()

    filename = model_name.lower().replace(" ", "_")
    plt.savefig(
        OUTPUT_DIR / f"{filename}_actual_vs_predicted.png",
        dpi=300
    )
    plt.show()
    plt.close()

    # 2. Average Actual vs Predicted Rainfall Trend
    prediction_df = test_df[["DIVISION", "YEAR"]].copy()
    prediction_df["Actual_Rainfall"] = y_test.to_numpy()
    prediction_df["Predicted_Rainfall"] = model_predictions

    yearly_actual = prediction_df.groupby("YEAR")["Actual_Rainfall"].mean()
    yearly_predicted = prediction_df.groupby("YEAR")["Predicted_Rainfall"].mean()

    plt.figure(figsize=(13, 6))

    plt.plot(
        yearly_actual.index,
        yearly_actual.values,
        marker="o",
        label="Actual"
    )

    plt.plot(
        yearly_predicted.index,
        yearly_predicted.values,
        marker="o",
        label="Predicted"
    )

    plt.title(f"Average Actual vs Predicted Annual Rainfall — {model_name}")
    plt.xlabel("Year")
    plt.ylabel("Average Annual Rainfall (mm)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{filename}_rainfall_trend.png",
        dpi=300
    )
    plt.show()
    plt.close()

    print(f"Graphs generated for {model_name}")

print("All remaining model graphs generated successfully!")


##Prediction error over time
prediction_df = test_df[["DIVISION", "YEAR"]].copy()
prediction_df["Actual_Rainfall"] = y_test.to_numpy()
prediction_df["Predicted_Rainfall"] = best_predictions
prediction_df["Error"] = (
    prediction_df["Actual_Rainfall"]
    - prediction_df["Predicted_Rainfall"]
)

print(prediction_df.head(10))

prediction_df.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False
)

yearly_actual = prediction_df.groupby("YEAR")["Actual_Rainfall"].mean()
yearly_predicted = prediction_df.groupby("YEAR")["Predicted_Rainfall"].mean()

plt.figure(figsize=(13, 6))
plt.plot(yearly_actual.index, yearly_actual.values, marker="o", label="Actual")
plt.plot(yearly_predicted.index, yearly_predicted.values, marker="o", label="Predicted")
plt.title(f"Average Actual vs Predicted Annual Rainfall — {best_model_name}")
plt.xlabel("Year")
plt.ylabel("Average Annual Rainfall (mm)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "rainfall_prediction_trend.png", dpi=300)
plt.show()




#Save the best model and final results
# Save the trained best model.
model_path = OUTPUT_DIR / "best_rainfall_model.joblib"
joblib.dump(best_model, model_path)

# Save final comparison and test predictions.
results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

prediction_df.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False
)

# Save a concise summary of the winning model.
best_row = results_df.iloc[0]

summary = f"""
RAINFALL PREDICTION PROJECT — RESULTS

Target: Annual rainfall
Training period: {train_df['YEAR'].min()}–{train_df['YEAR'].max()}
Testing period: {test_df['YEAR'].min()}–{test_df['YEAR'].max()}

Best model by RMSE: {best_model_name}
MAE: {best_row['MAE']:.3f}
RMSE: {best_row['RMSE']:.3f}
R²: {best_row['R2']:.4f}

Number of training records: {len(X_train)}
Number of testing records: {len(X_test)}

Note: Results apply to this dataset and chronological test period.
They do not guarantee future real-world performance.
"""

print(summary)

with open(OUTPUT_DIR / "project_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print("Best model saved to:", model_path.resolve())
print("All results saved to:", OUTPUT_DIR.resolve())
