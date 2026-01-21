"""
Rename videos to standardized format and create ground truth folders
Format: <subject>_<number>_<language>.mp4
"""
import re
import sys
from pathlib import Path
import shutil

raw_videos = Path('data/raw_videos')
ground_truth_base = Path('data/ground_truth')
ground_truth_base.mkdir(parents=True, exist_ok=True)

# Current videos (excluding test videos)
videos = [f for f in raw_videos.glob('*.mp4') if not f.name.startswith('input_video')]

print("Current videos:")
for v in sorted(videos):
    print(f"  {v.name}")

# Parse and standardize naming
def parse_video_name(filepath):
    """Extract subject, number, language from various formats."""
    name = filepath.stem.lower()
    
    # Patterns to extract info
    subject = 'unknown'
    number = '00'
    language = 'unknown'
    
    # Extract subject
    if 'chem' in name:
        subject = 'chemistry'
    elif 'phy' in name:
        subject = 'physics'
    elif 'math' in name:
        subject = 'mathematics'
    elif 'cn' in name or 'comp' in name or 'network' in name:
        subject = 'computer_networks'
    elif 'db' in name or 'database' in name:
        subject = 'database'
    elif 'daa' in name or 'algorithm' in name:
        subject = 'algorithms'
    elif 'english' in name:
        # Check if it has a subject after
        if 'chem' in name:
            subject = 'chemistry'
            language = 'english'
        elif 'phy' in name:
            subject = 'physics'
            language = 'english'
        elif 'math' in name:
            subject = 'mathematics'
            language = 'english'
    
    # Extract language if not set
    if language == 'unknown':
        if 'hindi' in name:
            language = 'hindi'
        elif 'english' in name or 'eng' in name:
            language = 'english'
    
    # Extract number
    numbers = re.findall(r'\d+', name)
    if numbers:
        number = f"{int(numbers[0]):02d}"
    
    return subject, number, language

print("\n" + "="*70)
print("PROPOSED RENAMING SCHEME")
print("="*70)

rename_map = {}
for video in sorted(videos):
    subject, number, language = parse_video_name(video)
    new_name = f"{subject}_{number}_{language}.mp4"
    rename_map[video.name] = new_name
    print(f"{video.name:30s} -> {new_name}")

print("\n" + "="*70)
print("CREATING GROUND TRUTH FOLDERS")
print("="*70)

for old_name, new_name in rename_map.items():
    video_id = new_name.replace('.mp4', '')
    gt_folder = ground_truth_base / video_id
    gt_folder.mkdir(parents=True, exist_ok=True)
    
    # Create placeholder timestamp file
    timestamp_file = gt_folder / 'transitions.txt'
    if not timestamp_file.exists():
        with open(timestamp_file, 'w') as f:
            f.write("# Transition timestamps (seconds)\n")
            f.write("# Format: one timestamp per line\n")
            f.write("# Example:\n")
            f.write("# 10.5\n")
            f.write("# 25.3\n")
            f.write("# 45.0\n")
    
    print(f"Created: {gt_folder}")

print("\n" + "="*70)
print("INSTRUCTIONS")
print("="*70)
print("""
1. For each video, add transition timestamps to:
   data/ground_truth/<video_id>/transitions.txt
   
2. Format: one timestamp per line (in seconds)
   Example:
   10.5
   25.3
   45.0
   
3. Run extraction with color mode:
   python main.py --video data/raw_videos/VIDEO.mp4 --output data --resize 640x360 --color-mode color
   
4. After adding timestamps, we can compare model vs ground truth
""")

# Optional: Actually rename files (commented out for safety)
print("\n" + "="*70)
print("RENAME FILES? (This will actually rename videos)")
print("="*70)

if '--execute' in sys.argv:
    print("EXECUTING RENAME...")
    for old_name, new_name in rename_map.items():
        old_path = raw_videos / old_name
        new_path = raw_videos / new_name
        if old_path.exists():
            old_path.rename(new_path)
            print(f"✓ Renamed: {old_name} -> {new_name}")
    print("\nAll files renamed successfully!")
else:
    print("To execute renaming, run:")
    print("  python rename_and_setup.py --execute")
    print("\nCurrently in DRY RUN mode - no files renamed")
