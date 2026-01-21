"""
Check which column has the actual transition labels
"""

import pandas as pd

df = pd.read_csv('labeled_dataset.csv', low_memory=False)

print("Checking all boolean/integer columns that might contain labels:\n")

for col in df.columns:
    if df[col].dtype in ['bool', 'int64', 'float64']:
        if df[col].nunique() <= 10:  # Binary or few values
            print(f"\nColumn: {col}")
            print(f"Value counts:\n{df[col].value_counts()}")
            print(f"Unique values: {df[col].unique()}")

# Also check the original data from toc_1
print("\n\n=== TOC_1 DATA ===")
toc1_data = df[df['video_name'] == 'toc_1']
print(f"toc_1 rows: {len(toc1_data)}")
print(f"toc_1 is_transition distribution:\n{toc1_data['is_transition'].value_counts()}")

if 'is_transition_gt' in df.columns:
    print(f"\ntoc_1 is_transition_gt distribution:\n{toc1_data['is_transition_gt'].value_counts()}")
