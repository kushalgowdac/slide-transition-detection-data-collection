"""
Analyze all videos in data/frames/ directory
Shows transitions detected for each video
"""
import pandas as pd
from pathlib import Path

def analyze_all_videos(output_dir='data', frames_dir='data/frames'):
    """Analyze all videos in frames directory."""
    
    frames_path = Path(frames_dir)
    videos = sorted([d.name for d in frames_path.iterdir() if d.is_dir()])
    
    print("\n" + "="*80)
    print("SUMMARY: ALL VIDEOS PROCESSED")
    print("="*80)
    
    results = []
    for video in videos:
        video_frames_dir = frames_path / video
        frame_count = len(list(video_frames_dir.glob('*.jpg')))
        results.append({'video': video, 'frames': frame_count})
    
    print(f"\nTotal videos: {len(results)}")
    for r in results:
        print(f"  {r['video']}: {r['frames']} frames extracted")
    
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    
    # Check if best_slides.csv exists
    best_path = Path(output_dir) / 'annotations' / 'best_slides.csv'
    if best_path.exists():
        best_df = pd.read_csv(best_path)
        video = best_df.iloc[0]['video_name']
        trans_count = best_df['transition_id'].nunique()
        
        print(f"\nLatest processed: {video}")
        print(f"Transitions detected: {trans_count}")
        print(f"Total candidates: {len(best_df)}")
        print(f"\nFirst 5 transitions:")
        
        for i, (tid, group) in enumerate(best_df.groupby('transition_id')):
            if i >= 5:
                break
            best = group[group['rank']==1].iloc[0]
            ts = best['timestamp']
            mins = int(ts // 60)
            secs = int(ts % 60)
            print(f"  {tid}: {mins}:{secs:02d}")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. For each video, run:
   python main.py --video data/raw_videos/VIDEO_NAME.mp4 --output data
   
2. Results saved to: data/annotations/best_slides.csv

3. Check quality by running:
   python analyze_results.py

4. Share with OCR/audio team:
   - data/annotations/best_slides.csv (timestamps + frame paths)
   - data/frames/VIDEO_NAME/ (actual images)
""")

if __name__ == '__main__':
    analyze_all_videos()
