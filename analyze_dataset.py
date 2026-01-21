"""
Analyze current labeled_dataset.csv to see toc_1 data
"""

import pandas as pd

df = pd.read_csv('labeled_dataset.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst rows:")
print(df.head())

print(f"\n\nLast rows:")
print(df.tail())

# Check for video identifiers
if 'video' in df.columns:
    print(f"\nUnique videos: {df['video'].nunique()}")
    print(df['video'].value_counts())
elif 'source_video' in df.columns:
    print(f"\nUnique source_video: {df['source_video'].nunique()}")
    print(df['source_video'].value_counts())

# Check for toc_1
print("\n\nSearching for toc_1 in dataset...")
for col in df.columns:
    if df[col].dtype == 'object':
        if df[col].astype(str).str.contains('toc', case=False, na=False).any():
            print(f"Found 'toc' in column '{col}':")
            print(df[df[col].astype(str).str.contains('toc', case=False, na=False)][col].unique())

# Check label distribution
if 'label' in df.columns:
    print(f"\n\nLabel distribution:")
    print(df['label'].value_counts())
elif 'is_transition' in df.columns:
    print(f"\n\nTransition distribution:")
    print(df['is_transition'].value_counts())
