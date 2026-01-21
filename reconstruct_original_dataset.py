"""
Reconstruct the original labeled_dataset.csv from annotation manifests.
This undoes the toc_1 merge that broke the model.
"""

import pandas as pd
import csv
import os
from pathlib import Path

# Read the train/val/test manifests to get original videos
manifest_dir = Path('data/annotations')

train_manifest = pd.read_csv(manifest_dir / 'train_manifest.csv')
val_manifest = pd.read_csv(manifest_dir / 'val_manifest.csv')
test_manifest = pd.read_csv(manifest_dir / 'test_manifest.csv')

print("Train manifest columns:", train_manifest.columns.tolist())
print("Train manifest shape:", train_manifest.shape)
print("\nFirst few rows:")
print(train_manifest.head())

print("\n\n=== ORIGINAL DATASET COMPOSITION ===")
print(f"Train videos: {len(train_manifest)}")
print(f"Val videos:   {len(val_manifest)}")
print(f"Test videos:  {len(test_manifest)}")

# Check what videos are in the original dataset
original_videos = set()
for df, name in [(train_manifest, 'train'), (val_manifest, 'val'), (test_manifest, 'test')]:
    for col in df.columns:
        if 'video' in col.lower() or 'file' in col.lower():
            videos = df[col].unique()
            print(f"\n{name.upper()} videos from '{col}':")
            for v in sorted(videos):
                print(f"  - {v}")
                original_videos.add(str(v))

print(f"\n\nTotal unique videos in original split: {len(original_videos)}")
print("Videos:", sorted(original_videos))

# Check if toc_1 is in original split
if any('toc' in str(v).lower() for v in original_videos):
    print("\n[WARNING] Found 'toc' in original videos - these shouldn't be there!")
else:
    print("\n[OK] toc_1 was not in original training set (as expected)")
