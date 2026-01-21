"""
Process videos that have ground truth timestamps
Shows progress and estimated time
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import time

def has_ground_truth(video_id):
    """Check if video has timestamps."""
    gt_file = Path('data/ground_truth') / video_id / 'transitions.txt'
    if not gt_file.exists():
        return False
    
    with open(gt_file) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                return True
    return False

def is_processed(video_id):
    """Check if video already processed."""
    output_dir = Path('data') / f'processed_{video_id}'
    metadata_file = output_dir / 'annotations' / 'frames_metadata.csv'
    checkpoint_file = output_dir / 'annotations' / '.extraction_complete'
    # Consider processed if both metadata exists and checkpoint is set
    return metadata_file.exists() and checkpoint_file.exists()

def main():
    gt_dir = Path('data/ground_truth')
    raw_videos_dir = Path('data/raw_videos')
    
    # Get videos with ground truth
    videos_to_process = []
    for gt_folder in sorted(gt_dir.iterdir()):
        if gt_folder.is_dir() and has_ground_truth(gt_folder.name):
            video_file = raw_videos_dir / f'{gt_folder.name}.mp4'
            if video_file.exists():
                videos_to_process.append((gt_folder.name, video_file))
    
    print("="*70)
    print(f"Found {len(videos_to_process)} videos with ground truth")
    print("RESUMABLE: If interrupted, will continue from last completed video")
    print("="*70)
    
    # Check already processed
    to_process = []
    for video_id, video_file in videos_to_process:
        if is_processed(video_id):
            print(f"✓ SKIP {video_id} (already processed)")
        else:
            to_process.append((video_id, video_file))
    
    if not to_process:
        print("\n✓ All videos already processed!")
        return
    
    print(f"\n{len(to_process)} videos remaining")
    print("="*70)
    
    start_time = time.time()
    
    for i, (video_id, video_file) in enumerate(to_process, 1):
        print(f"\n[{i}/{len(to_process)}] Processing: {video_id}")
        print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
        
        video_start = time.time()
        
        output_dir = Path('data') / f'processed_{video_id}'
        
        cmd = [
            sys.executable,
            'main.py',
            '--video', str(video_file),
            '--output', str(output_dir),
            '--fps', '1.0',
            '--resize', '640x360',
            '--color-mode', 'color',
            '--edge-threshold', '4.0',
            '--dense-threshold', '0.3',
        ]
        
        result = subprocess.run(cmd)
        
        video_elapsed = time.time() - video_start
        
        if result.returncode != 0:
            print(f"✗ ERROR processing {video_id}")
        else:
            print(f"✓ Completed in {video_elapsed/60:.1f} minutes")
        
        # Estimate remaining time
        if i < len(to_process):
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = avg_time * (len(to_process) - i)
            print(f"Estimated remaining: {remaining/60:.1f} minutes")
    
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print(f"✓ Batch processing complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Processed: {len(to_process)} videos")

if __name__ == '__main__':
    main()
