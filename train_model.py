
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    classification_report
)

# 1. Load original historical dataset
df = pd.read_csv("weatherHistory.csv")

# 2. Convert date column into datetime
df["Formatted Date"] = pd.to_datetime(
    df["Formatted Date"],
    utc=True,
    errors="coerce"
)

# 3. Sort records chronologically
df = df.dropna(subset=["Formatted Date"])
df = df.sort_values("Formatted Date").reset_index(drop=True)

# 4. Extract date and time features
df["hour"] = df["Formatted Date"].dt.hour
df["month"] = df["Formatted Date"].dt.month
df["day"] = df["Formatted Date"].dt.day
df["day_of_week"] = df["Formatted Date"].dt.dayofweek

# 5. Create next-hour prediction targets
df["next_temperature"] = df["Temperature (C)"].shift(-1)
df["next_precipitation"] = df["Precip Type"].shift(-1)
df["next_weather"] = df["Summary"].shift(-1)

# 6. Select input features
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

# 7. Separate numerical and categorical features
numeric_features = [
    "Humidity",
    "Apparent Temperature (C)",
    "Wind Speed (km/h)",
    "Wind Bearing (degrees)",
    "Visibility (km)",
    "Pressure (millibars)",
    "hour",
    "month",
    "day",
    "day_of_week"
]

categorical_features = [
    "Summary",
    "Precip Type"
]

# 8. Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 9. Create a function for chronological splitting
def prepare_data(target):
    data = df.dropna(subset=[target]).copy()

    X = data[features]
    y = data[target]

    split = int(len(data) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    return X_train, X_test, y_train, y_test


# 10. Temperature prediction model
print("\nTRAINING TEMPERATURE MODEL")

X_train, X_test, y_train, y_test = prepare_data(
    "next_temperature"
)

temperature_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

temperature_model.fit(X_train, y_train)

temperature_predictions = temperature_model.predict(X_test)

print("Temperature MAE:",
      mean_absolute_error(y_test, temperature_predictions))

print("Temperature RMSE:",
      np.sqrt(mean_squared_error(y_test, temperature_predictions)))

joblib.dump(temperature_model, "temperature_model.pkl")


# 11. Precipitation prediction model
print("\nTRAINING PRECIPITATION MODEL")

X_train, X_test, y_train, y_test = prepare_data(
    "next_precipitation"
)

precipitation_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ))
])

precipitation_model.fit(X_train, y_train)

precipitation_predictions = precipitation_model.predict(X_test)

print("Precipitation Accuracy:",
      accuracy_score(y_test, precipitation_predictions))

print(classification_report(
    y_test,
    precipitation_predictions,
    zero_division=0
))

joblib.dump(precipitation_model, "precipitation_model.pkl")


# 12. Weather condition prediction model
print("\nTRAINING WEATHER CONDITION MODEL")

X_train, X_test, y_train, y_test = prepare_data(
    "next_weather"
)

weather_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ))
])

weather_model.fit(X_train, y_train)

weather_predictions = weather_model.predict(X_test)

print("Weather Condition Accuracy:",
      accuracy_score(y_test, weather_predictions))

print(classification_report(
    y_test,
    weather_predictions,
    zero_division=0
))

joblib.dump(weather_model, "weather_model.pkl")

print("\nALL THREE MODELS TRAINED SUCCESSFULLY!")
print("Models saved successfully!")
