# Dataset Organization Guide

## Video Naming Convention

All videos follow the format: `<subject>_<number>_<language>.mp4`

Examples:
- `chemistry_01_english.mp4`
- `mathematics_05_hindi.mp4`
- `physics_01_english.mp4`

## Directory Structure

```
data/
├── raw_videos/               # Original lecture videos
│   ├── chemistry_01_english.mp4
│   ├── chemistry_04_english.mp4
│   ├── ...
│
├── ground_truth/            # Manual transition timestamps
│   ├── chemistry_01_english/
│   │   └── transitions.txt
│   ├── chemistry_04_english/
│   │   └── transitions.txt
│   └── ...
│
└── processed_<video_id>/    # Extracted frames & metadata
    ├── frames/
    │   └── <video_id>/
    │       ├── frame_0001.jpg
    │       ├── frame_0002.jpg
    │       └── ...
    └── annotations/
        ├── frames_metadata.csv
        ├── best_slides.csv
        └── annotation_manifest.csv
```

## Ground Truth Format

Each video has a `transitions.txt` file with transition and optional ideal frame timestamps.

**Enhanced format** (with ideal frame timestamps):
```
# transitions.txt
10.5 | 8.3
25.3 | 23.1
45.0 | 44.2
67.2 | 65.8
```
Format: `transition_time | ideal_frame_time`
- `transition_time`: When the slide actually changes
- `ideal_frame_time`: Best frame to capture (full content, no occlusion)

**Simple format** (transitions only):
```
# transitions.txt
10.5
25.3
45.0
67.2
```

**Rules:**
- Timestamps in seconds (float)
- Lines starting with `#` are comments
- Blank lines are ignored
- Ideal frame should be 1-10 seconds before transition

## Workflow

### 1. Add Ground Truth Timestamps

For each video in `data/raw_videos/`:
1. Watch the video
2. Note the timestamp when each slide transition occurs
3. Add timestamps to `data/ground_truth/<video_id>/transitions.txt`

Example:
```bash
# Edit ground truth file
notepad data/ground_truth/chemistry_01_english/transitions.txt
```

### 2. Process Videos

**Single video:**
```bash
python main.py --video data/raw_videos/chemistry_01_english.mp4 \
               --output data/processed_chemistry_01_english \
               --color-mode color \
               --resize 640x360 \
               --edge-threshold 4.0
```

**Batch processing:**
```bash
python batch_process.py --color-mode color --edge-threshold 4.0
```

### 3. Validate Results

Compare model predictions vs ground truth:
```bash
# Validate all videos
python validate_ground_truth.py --tolerance 5.0

# Validate specific video
python validate_ground_truth.py --video chemistry_01_english --tolerance 5.0
```

This will show:
- Recall (% of ground truth transitions detected)
- Precision (% of predictions that are true transitions)
- Matched, missed, and false positive transitions

### 4. Analyze Results

Quick analysis of latest extraction:
```bash
python analyze_results.py
```

List all processed videos:
```bash
python summary.py
```

## Video Specifications

- Format: MP4
- Resolution: Portrait 360x640 (some may be landscape)
- Color: Full color (RGB)
- Content: PW-style PowerPoint/smartboard lectures
- **Note:** Whiteboard instant-erase lectures are not supported

## Next Steps

After collecting ground truth for all videos:
1. Run batch processing: `python batch_process.py`
2. Validate results: `python validate_ground_truth.py`
3. Prepare training dataset (combine all metadata with ground truth labels)
4. Train ML classifier: `python src/classifier.py train`
5. Integrate trained model into inference pipeline

## Current Video Inventory

14 PW lecture videos:
- Chemistry: 4 Hindi, 4 English (8 total)
- Physics: 1 Hindi, 2 English (3 total)
- Mathematics: 2 Hindi, 1 English (3 total)
- Computer Networks: 1 Hindi
- Database: 2 Hindi
- Algorithms: 1 Hindi

Plus 4 test videos for benchmarking.
