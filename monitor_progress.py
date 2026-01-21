"""
Monitor pipeline progress and notify on completion
"""
import time
from pathlib import Path
import subprocess
import sys
from datetime import datetime

def count_completed_videos():
    """Count how many videos are fully processed."""
    data_dir = Path('data')
    completed = 0
    
    for proc_dir in data_dir.glob('processed_*/'):
        checkpoint = proc_dir / 'annotations' / '.extraction_complete'
        if checkpoint.exists():
            completed += 1
    
    return completed

def monitor_processing(total_videos=14):
    """Monitor video processing progress."""
    print("="*70)
    print("MONITORING VIDEO PROCESSING")
    print("="*70)
    
    last_count = 0
    start_time = time.time()
    
    while True:
        current_count = count_completed_videos()
        
        if current_count > last_count:
            elapsed = time.time() - start_time
            avg_time = elapsed / current_count if current_count > 0 else 0
            remaining_est = avg_time * (total_videos - current_count)
            
            print(f"\n🎉 Video {current_count}/{total_videos} COMPLETE!")
            print(f"   Elapsed: {elapsed/60:.1f} min")
            print(f"   Estimated remaining: {remaining_est/60:.1f} min")
            
            last_count = current_count
        
        if current_count >= total_videos:
            print(f"\n{'='*70}")
            print("✅ ALL VIDEOS PROCESSED!")
            print(f"{'='*70}")
            print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
            break
        
        # Check every 30 seconds
        time.sleep(30)

if __name__ == '__main__':
    monitor_processing()
