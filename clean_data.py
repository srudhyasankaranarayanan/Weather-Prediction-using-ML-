
import pandas as pd

# 1. Load the dataset
df = pd.read_csv("weatherHistory.csv")

# 2. Remove duplicate rows
df = df.drop_duplicates()

# 3. Display missing values before cleaning
print("Missing values before cleaning:")
print(df.isnull().sum())

# 4. Display dataset information
print("\nDataset information:")
df.info()

# 5. Separate numerical and text columns
numeric_columns = df.select_dtypes(include="number").columns
text_columns = df.select_dtypes(include="object").columns

# 6. Fill missing numerical values with the median
for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# 7. Fill missing text values with the most frequent value
for column in text_columns:
    if df[column].isnull().any() and not df[column].mode().empty:
        df[column] = df[column].fillna(df[column].mode()[0])

# 8. Display missing values after cleaning
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# 9. Display cleaned dataset size
print("\nCleaned dataset shape:")
print(df.shape)

# 10. Save the cleaned dataset
df.to_csv("weather_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")
