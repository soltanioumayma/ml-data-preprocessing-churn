import pandas as pd
import os

# List of CSV files to check
csv_files = [
    "1) iris.csv",
    "3) Sentiment dataset.csv", 
    "4) house Prediction Data Set.csv",
    "churn-bigml-20.csv",
    "churn-bigml-80.csv"
]

print("Checking for missing data in each dataset:\n")
print("=" * 60)

for file in csv_files:
    if os.path.exists(file):
        try:
            df = pd.read_csv(file)
            missing_count = df.isnull().sum().sum()
            total_cells = df.shape[0] * df.shape[1]
            missing_percentage = (missing_count / total_cells) * 100 if total_cells > 0 else 0
            
            print(f"\nFile: {file}")
            print(f"Shape: {df.shape}")
            print(f"Total missing values: {missing_count}")
            print(f"Missing percentage: {missing_percentage:.2f}%")
            
            if missing_count > 0:
                print("\nMissing values by column:")
                missing_cols = df.isnull().sum()
                missing_cols = missing_cols[missing_cols > 0]
                print(missing_cols)
            else:
                print("No missing data found")
                
        except Exception as e:
            print(f"\nError reading {file}: {e}")
    else:
        print(f"\nFile not found: {file}")

print("\n" + "=" * 60)
