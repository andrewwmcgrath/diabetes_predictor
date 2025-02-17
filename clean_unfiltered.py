import pandas as pd
import numpy as np
import warnings
from sklearn.impute import SimpleImputer
import os
import re

warnings.filterwarnings("ignore")
directory = os.getcwd()
input_file_path = os.path.join(directory, 'balanced_data.csv') 
output_file_path = os.path.join(directory, 'cleaned.csv')

df = pd.read_csv(input_file_path)
print(f"Data loaded successfully with shape: {df.shape}")

diabetics = df['2443-0.0_Diabetes diagnosed by doctor']
mean = diabetics.mean()
print(f'mean1 = {mean}')


diabetics = df['2443-0.0_Diabetes diagnosed by doctor']
mean = diabetics.mean()
print(f'mean2 = {mean}')

diabetics = df['2443-0.0_Diabetes diagnosed by doctor']
mean = diabetics.mean()
print(f'mean3 = {mean}')

date_data = []
date_pattern = re.compile(r'^\d{1,2}/\d{1,2}/(\d{2}|\d{4})$')

for col in df.columns:
    no_nans = df[col].dropna()
    if not no_nans.empty:
        if no_nans.apply(lambda x: bool(date_pattern.match(str(x)))).any():
            date_data.append(col)

print(f"First 5 date columns (proof they exist): {date_data[:5]}")

def convert_to_numeric_dates(df, date_cols):
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce').astype('int64') // (10**9 * 60 * 60 * 24)
    return df

df = convert_to_numeric_dates(df, date_data)

df.replace('', pd.NA, inplace=True)
df.replace(['None', 'N/A', 'null'], pd.NA, inplace=True)

diabetics = df['2443-0.0_Diabetes diagnosed by doctor']
mean = diabetics.mean()
print(f'mean4 = {mean}')

# Ensure all NaN values are consistent
df = df.apply(pd.to_numeric, errors='coerce')

# List to store non-numeric columns
non_numeric_columns = []

# Identify non-numeric columns
for col in df.columns:
    if df[col].dtype == 'object' or not pd.api.types.is_numeric_dtype(df[col]):
        non_numeric_columns.append(col)

# Output the non-numeric columns
print("\nNon-numeric columns identified, here are the first 5:")
print(non_numeric_columns[:5])
print(f'there are {len(non_numeric_columns)} non numerical columns')

diabetics = df['2443-0.0_Diabetes diagnosed by doctor']
mean = diabetics.mean()
print(f'mean5 = {mean}')

df.dropna(thresh= 0.95 * len(df), axis=1, inplace=True)

numeric_cols = df.columns
num_imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

diabetics = df['2443-0.0_Diabetes diagnosed by doctor']
mean = diabetics.mean()
print(f'mean6 = {mean}')

# Check for NaN values
nan_summary = df.isnull().sum()
nan_columns = nan_summary[nan_summary > 0]

if nan_columns.empty:
    print("No NaN values remaining. Can now test data.")
else:
    print(f"Found NaN values in {len(nan_columns)} columns.")
    print("Columns with NaN values:")
    for col, count in nan_columns.items():
        print(f"{col}: {count} NaN values")
    
    # Additional details (optional)
    print("\nDetailed NaN Summary:")
    print(nan_columns)

# Save the cleaned data
df.to_csv(output_file_path, index=False)
print(f"Cleaned dataset has been written to: {output_file_path} with shape {df.shape}")