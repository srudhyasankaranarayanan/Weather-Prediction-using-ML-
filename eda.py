
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned weather data
df = pd.read_csv("weather_cleaned.csv")

# Display available columns
print("Available columns:")
print(df.columns.tolist())

# Display basic statistical information
print("\nStatistical summary:")
print(df.describe())

# Select numerical columns
numeric_df = df.select_dtypes(include="number")

# 1. Distribution of numerical features
numeric_df.hist(figsize=(12, 8), bins=20)
plt.suptitle("Weather Data Distributions")
plt.tight_layout()
plt.show()

# 2. Correlation heatmap
if numeric_df.shape[1] >= 2:
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title("Correlation Between Weather Features")
    plt.tight_layout()
    plt.show()

# 3. Temperature distribution, if available
temperature_columns = [
    col for col in df.columns
    if "temperature" in col.lower() or col.lower() in ["temp", "maxtemp", "mintemp"]
]

if temperature_columns:
    column = temperature_columns[0]
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.tight_layout()
    plt.show()
else:
    print("\nNo recognizable temperature column found.")

print("\nEDA completed!")
