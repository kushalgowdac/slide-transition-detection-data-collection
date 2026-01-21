"""
Create labeled dataset from processed videos and ground truth
Resumable: Saves progress checkpoints
"""
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

def load_ground_truth(video_id):
    """Load ground truth timestamps."""
    gt_file = Path('data/ground_truth') / video_id / 'transitions.txt'
    
    if not gt_file.exists():
        return None, None
    
    transitions = []
    ideal_frames = []
    
    with open(gt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            try:
                if '|' in line:
                    parts = line.split('|')
                    transition_time = float(parts[0].strip())
                    ideal_time = float(parts[1].strip())
                    transitions.append(transition_time)
                    ideal_frames.append(ideal_time)
                else:
                    transitions.append(float(line))
            except ValueError:
                continue
    
    return sorted(transitions), sorted(ideal_frames) if ideal_frames else None

def is_near_transition(timestamp, transitions, window=5.0):
    """Check if timestamp is within window of any transition."""
    for t in transitions:
        if abs(timestamp - t) <= window:
            return True
    return False

def create_labeled_dataset():
    """Create labeled dataset from all processed videos."""
    
    checkpoint_file = Path('dataset_creation_checkpoint.json')
    output_file = Path('labeled_dataset.csv')
    
    # Check if already completed
    if output_file.exists():
        print(f"✓ Dataset already exists: {output_file}")
        response = input("Recreate dataset? (y/n): ")
        if response.lower() != 'y':
            print("Using existing dataset.")
            return output_file
    
    print("="*70)
    print("CREATING LABELED DATASET")
    print("="*70)
    
    gt_dir = Path('data/ground_truth')
    processed_base = Path('data')
    
    # Get all videos with ground truth
    video_ids = [d.name for d in gt_dir.iterdir() if d.is_dir()]
    
    all_data = []
    processed_videos = []
    
    for i, video_id in enumerate(sorted(video_ids), 1):
        print(f"\n[{i}/{len(video_ids)}] Processing {video_id}...")
        
        # Load ground truth
        gt_transitions, gt_ideal = load_ground_truth(video_id)
        
        if not gt_transitions:
            print(f"  ⚠ No ground truth found, skipping")
            continue
        
        # Load processed frames
        processed_dir = processed_base / f'processed_{video_id}'
        metadata_file = processed_dir / 'annotations' / 'frames_metadata.csv'
        
        if not metadata_file.exists():
            print(f"  ⚠ Not processed yet, skipping")
            continue
        
        frames_df = pd.read_csv(metadata_file)
        
        # Label frames based on ground truth
        frames_df['video_id'] = video_id
        frames_df['is_transition_gt'] = frames_df['timestamp'].apply(
            lambda t: is_near_transition(t, gt_transitions, window=5.0)
        )
        
        # Add ground truth transition IDs
        frames_df['transition_id_gt'] = -1
        for idx, trans_time in enumerate(gt_transitions):
            mask = (frames_df['timestamp'] >= trans_time - 5) & (frames_df['timestamp'] <= trans_time + 5)
            frames_df.loc[mask, 'transition_id_gt'] = idx
        
        print(f"  ✓ {len(frames_df)} frames, {gt_transitions} transitions")
        print(f"  Positive samples: {frames_df['is_transition_gt'].sum()}")
        
        all_data.append(frames_df)
        processed_videos.append(video_id)
        
        # Save checkpoint periodically
        if i % 3 == 0:
            checkpoint_data = {
                'processed_videos': processed_videos,
                'timestamp': datetime.now().isoformat(),
                'progress': f"{i}/{len(video_ids)}"
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            print(f"  💾 Checkpoint saved ({i}/{len(video_ids)})")
    
    if not all_data:
        print("\n⚠ No data to create dataset!")
        return None
    
    # Combine all videos
    print("\n" + "="*70)
    print("COMBINING DATA")
    print("="*70)
    
    dataset = pd.concat(all_data, ignore_index=True)
    
    print(f"Total frames: {len(dataset)}")
    print(f"Total videos: {len(processed_videos)}")
    print(f"Positive samples: {dataset['is_transition_gt'].sum()} ({dataset['is_transition_gt'].sum()/len(dataset)*100:.1f}%)")
    print(f"Negative samples: {(~dataset['is_transition_gt']).sum()} ({(~dataset['is_transition_gt']).sum()/len(dataset)*100:.1f}%)")
    
    # Save dataset
    dataset.to_csv(output_file, index=False)
    print(f"\n✓ Dataset saved: {output_file}")
    
    # Save metadata
    metadata = {
        'created': datetime.now().isoformat(),
        'videos': processed_videos,
        'total_frames': len(dataset),
        'positive_samples': int(dataset['is_transition_gt'].sum()),
        'negative_samples': int((~dataset['is_transition_gt']).sum()),
        'video_count': len(processed_videos)
    }
    
    metadata_file = Path('labeled_dataset_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved: {metadata_file}")
    
    # Create train/val/test splits by video
    print("\n" + "="*70)
    print("CREATING SPLITS")
    print("="*70)
    
    from sklearn.model_selection import train_test_split
    
    # Split by video to avoid data leakage
    train_videos, test_videos = train_test_split(processed_videos, test_size=0.2, random_state=42)
    train_videos, val_videos = train_test_split(train_videos, test_size=0.15, random_state=42)
    
    dataset['split'] = 'train'
    dataset.loc[dataset['video_id'].isin(val_videos), 'split'] = 'val'
    dataset.loc[dataset['video_id'].isin(test_videos), 'split'] = 'test'
    
    print(f"Train videos: {len(train_videos)} ({dataset[dataset['split']=='train'].shape[0]} frames)")
    print(f"Val videos:   {len(val_videos)} ({dataset[dataset['split']=='val'].shape[0]} frames)")
    print(f"Test videos:  {len(test_videos)} ({dataset[dataset['split']=='test'].shape[0]} frames)")
    
    # Save with splits
    dataset.to_csv(output_file, index=False)
    
    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    
    return output_file

if __name__ == '__main__':
    dataset_file = create_labeled_dataset()
    
    if dataset_file:
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("1. Review dataset: python -c \"import pandas as pd; print(pd.read_csv('labeled_dataset.csv').describe())\"")
        print("2. Train model: python train_model.py")
