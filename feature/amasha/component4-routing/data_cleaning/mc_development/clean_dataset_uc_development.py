import pandas as pd

# 1. Load the Excel file
file_path = r'D:\SLIIT\Y4S1\Research\dataset_formation\uc_development_original.xlsx'
df = pd.read_excel(file_path)

# 2. Remove rows where ANY column has a null/NaN value
cleaned_df = df.dropna()

print(f"Row count after removing nulls: {len(cleaned_df)}")

# 3. Save the cleaned dataframe to a new Excel file
output_path = r'D:\SLIIT\Y4S1\Research\dataset_formation\uc_development_null_removed.xlsx'
cleaned_df.to_excel(output_path, index=False)

print(f"Cleaned file successfully saved to {output_path}")