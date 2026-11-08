
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load original dataset
df = pd.read_csv("weatherHistory.csv")

# Convert date
df["Formatted Date"] = pd.to_datetime(
    df["Formatted Date"],
    utc=True,
    errors="coerce"
)

df = df.dropna(subset=["Formatted Date"])
df = df.sort_values("Formatted Date").reset_index(drop=True)

# Extract time features
df["hour"] = df["Formatted Date"].dt.hour
df["month"] = df["Formatted Date"].dt.month
df["day"] = df["Formatted Date"].dt.day
df["day_of_week"] = df["Formatted Date"].dt.dayofweek

# Create targets
df["next_temperature"] = df["Temperature (C)"].shift(-1)
df["next_precipitation"] = df["Precip Type"].shift(-1)
df["next_weather"] = df["Summary"].shift(-1)

# Input features
features = [
    "Humidity",
    "Apparent Temperature (C)",
    "Wind Speed (km/h)",
    "Wind Bearing (degrees)",
    "Visibility (km)",
    "Pressure (millibars)",
    "Summary",
    "Precip Type",
    "hour",
    "month",
    "day",
    "day_of_week"
]

# Evaluate each model using its chronological test portion
targets = {
    "Temperature": "next_temperature",
    "Precipitation": "next_precipitation",
    "Weather Condition": "next_weather"
}

for name, target in targets.items():

    data = df.dropna(subset=[target]).copy()

    X = data[features]
    y = data[target]

    split = int(len(data) * 0.8)

    X_test = X.iloc[split:]
    y_test = y.iloc[split:]

    # Load corresponding model
    if name == "Temperature":
        model = joblib.load("temperature_model.pkl")

    elif name == "Precipitation":
        model = joblib.load("precipitation_model.pkl")

    else:
        model = joblib.load("weather_model.pkl")

    predictions = model.predict(X_test)

    print("\n" + "=" * 45)
    print(name.upper(), "MODEL EVALUATION")
    print("=" * 45)

    if name == "Temperature":

        mae = mean_absolute_error(y_test, predictions)

        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )

        print("MAE:", round(mae, 3), "°C")
        print("RMSE:", round(rmse, 3), "°C")

    else:

        accuracy = accuracy_score(y_test, predictions)

        print("Accuracy:", round(accuracy * 100, 2), "%")

        print("\nClassification Report:")
        print(classification_report(
            y_test,
            predictions,
            zero_division=0
        ))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))

print("\nEvaluation completed!")
