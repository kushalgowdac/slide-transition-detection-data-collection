"""
Add train/val/test splits to labeled dataset
"""
import pandas as pd
from pathlib import Path

# Load dataset
df = pd.read_csv('labeled_dataset.csv')

print("Adding train/val/test splits...")

# Get unique videos
videos = df['video_id'].unique()
print(f"Total videos: {len(videos)}")

# Manual split (70% train, 15% val, 15% test)
train_videos = videos[:10]  # First 10
val_videos = videos[10:12]  # Next 2
test_videos = videos[12:]   # Last 2

df['split'] = 'train'
df.loc[df['video_id'].isin(val_videos), 'split'] = 'val'
df.loc[df['video_id'].isin(test_videos), 'split'] = 'test'

print(f"\nTrain videos: {list(train_videos)}")
print(f"Val videos:   {list(val_videos)}")
print(f"Test videos:  {list(test_videos)}")

print(f"\nTrain: {len(df[df['split']=='train'])} frames")
print(f"Val:   {len(df[df['split']=='val'])} frames")
print(f"Test:  {len(df[df['split']=='test'])} frames")

# Save
df.to_csv('labeled_dataset.csv', index=False)
print("\n✓ Dataset splits added!")
print("✓ labeled_dataset.csv saved")
