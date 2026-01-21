"""
Restore the original 14-video dataset by removing toc_1 data
"""

import pandas as pd
import numpy as np

# Load the corrupted dataset
print("Loading current dataset...")
df = pd.read_csv('labeled_dataset.csv', low_memory=False)
print(f"Current shape: {df.shape}")
print(f"Videos: {df['video_name'].unique()}")

# Remove toc_1 data
print("\nRemoving toc_1 data...")
df_original = df[df['video_name'] != 'toc_1'].reset_index(drop=True)
print(f"After removing toc_1: {df_original.shape}")
print(f"Remaining videos: {df_original['video_name'].unique()}")

# Save as original dataset
output_file = 'labeled_dataset_original_14videos.csv'
df_original.to_csv(output_file, index=False)
print(f"\nSaved to: {output_file}")

# Check label distribution  
print(f"\nLabel distribution in restored dataset:")
if 'is_transition' in df_original.columns:
    print(df_original['is_transition'].value_counts())

if 'is_transition_gt' in df_original.columns:
    print("\nis_transition_gt distribution:")
    print(df_original['is_transition_gt'].value_counts())
