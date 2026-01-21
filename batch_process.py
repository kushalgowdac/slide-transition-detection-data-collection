"""
Batch process all videos in raw_videos directory
Extracts frames with colored output for portrait videos
"""
import argparse
from pathlib import Path
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description='Batch process lecture videos')
    parser.add_argument('--fps', type=float, default=1.0, help='Frames per second to extract')
    parser.add_argument('--resize', type=str, default='640x360', help='Resize frames (WxH)')
    parser.add_argument('--color-mode', type=str, default='color', choices=['color', 'gray'], help='Color mode')
    parser.add_argument('--edge-threshold', type=float, default=4.0, help='Edge change threshold for transitions')
    parser.add_argument('--dense-threshold', type=float, default=0.3, help='Dense sampling threshold')
    parser.add_argument('--skip-existing', action='store_true', help='Skip videos with existing output')
    args = parser.parse_args()
    
    raw_videos_dir = Path('data/raw_videos')
    output_base = Path('data')
    
    # Get all videos (exclude test videos if needed)
    videos = sorted([v for v in raw_videos_dir.glob('*.mp4') if not v.name.startswith('input_video')])
    
    if not videos:
        print("No videos found in data/raw_videos/")
        return
    
    print(f"Found {len(videos)} videos to process")
    print("="*70)
    
    for i, video in enumerate(videos, 1):
        video_id = video.stem
        output_dir = output_base / f'processed_{video_id}'
        
        # Check if already processed
        if args.skip_existing and output_dir.exists():
            print(f"[{i}/{len(videos)}] SKIP {video.name} (output exists)")
            continue
        
        print(f"\n[{i}/{len(videos)}] Processing: {video.name}")
        print(f"Output: {output_dir}")
        
        # Build command
        cmd = [
            sys.executable,
            'main.py',
            '--video', str(video),
            '--output', str(output_dir),
            '--fps', str(args.fps),
            '--resize', args.resize,
            '--color-mode', args.color_mode,
            '--edge-threshold', str(args.edge_threshold),
            '--dense-threshold', str(args.dense_threshold),
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        # Run extraction
        result = subprocess.run(cmd, capture_output=False)
        
        if result.returncode != 0:
            print(f"ERROR processing {video.name}")
        else:
            print(f"✓ Completed {video.name}")
    
    print("\n" + "="*70)
    print("Batch processing complete!")
    print("\nNext steps:")
    print("1. Add transition timestamps to data/ground_truth/<video_id>/transitions.txt")
    print("2. Run validation: python validate_ground_truth.py")
    print("3. Train classifier: python src/classifier.py train")

if __name__ == '__main__':
    main()
