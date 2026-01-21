"""
Quick check of batch processing progress
"""
from pathlib import Path

processed_dir = Path('data')
ground_truth_dir = Path('data/ground_truth')

# Get all videos that should be processed
videos_to_process = [d.name for d in ground_truth_dir.iterdir() if d.is_dir()]

print("="*70)
print("BATCH PROCESSING PROGRESS")
print("="*70)

completed = 0
pending = 0

for video_id in sorted(videos_to_process):
    output_dir = processed_dir / f'processed_{video_id}'
    metadata_file = output_dir / 'annotations' / 'frames_metadata.csv'
    
    if metadata_file.exists():
        status = "✓ COMPLETE"
        completed += 1
    else:
        status = "⏳ PENDING"
        pending += 1
    
    print(f"{status:12s} {video_id}")

print("="*70)
print(f"Completed: {completed}/{len(videos_to_process)}")
print(f"Pending:   {pending}/{len(videos_to_process)}")
print(f"Progress:  {completed/len(videos_to_process)*100:.1f}%")
