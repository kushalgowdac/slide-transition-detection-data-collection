"""
Quick analysis of extraction results
Run after main.py completes to get summary of transitions detected
Usage: python analyze_results.py
"""
import pandas as pd
from pathlib import Path

def analyze_video(output_dir='data/annotations'):
    """Analyze latest extraction results."""
    
    output_path = Path(output_dir)
    
    # Load frames
    frames_path = output_path / 'frames_metadata.csv'
    if not frames_path.exists():
        print("ERROR: frames_metadata.csv not found. Run main.py first!")
        return
    
    frames_df = pd.read_csv(frames_path)
    
    if frames_df.empty:
        print("No frames found!")
        return
    
    video_name = frames_df.iloc[0]['video_name']
    duration = frames_df['timestamp'].max()
    
    print("\n" + "="*70)
    print(f"EXTRACTION RESULTS: {video_name}")
    print("="*70)
    print(f"Duration: {duration:.0f}s ({duration/60:.1f} minutes)")
    print(f"Total frames extracted: {len(frames_df)}")
    print(f"Transitions detected: {len(frames_df[frames_df['is_transition']==True])}")
    
    # Load best slides
    best_path = output_path / 'best_slides.csv'
    if best_path.exists():
        best_df = pd.read_csv(best_path)
        print(f"\nBest slide candidates: {len(best_df)}")
        print(f"Unique transitions: {best_df['transition_id'].nunique()}")
        
        print("\n" + "="*70)
        print("TRANSITIONS TIMELINE")
        print("="*70)
        
        for tid, group in best_df.groupby('transition_id'):
            first = group.iloc[0]
            timestamp = first['timestamp']
            mins = int(timestamp // 60)
            secs = int(timestamp % 60)
            quality = first['frame_quality']
            fullness = first['content_fullness']
            
            print(f"\n{tid}:")
            print(f"  Time: {mins}:{secs:02d} (best candidate)")
            print(f"  Quality: {quality:.3f}, Fullness: {fullness:.1%}")
            print(f"  Candidates: {len(group)} frames to choose from")
    else:
        print("\nNo best_slides.csv found. Check main.py output.")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    analyze_video()
