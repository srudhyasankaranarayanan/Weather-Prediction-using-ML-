
import pandas as pd

# Load the weather dataset
df = pd.read_csv("weatherHistory.csv")

# Display the first five records
print("First five rows:")
print(df.head())

# Display the column names
print("\nColumn names:")
print(df.columns.tolist())

# Display the number of rows and columns
print("\nDataset shape:")
print(df.shape)

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Display data types
print("\nData types:")
print(df.dtypes)
