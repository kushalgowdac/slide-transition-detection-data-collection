# Quick Start Guide - Dataset Collection

## ✅ What's Been Done

1. **Renamed 14 PW lecture videos** to standardized format:
   - Format: `<subject>_<number>_<language>.mp4`
   - Examples: `chemistry_01_english.mp4`, `mathematics_05_hindi.mp4`
   - Location: `data/raw_videos/`

2. **Created ground truth folder structure**:
   - 14 folders in `data/ground_truth/`
   - Each contains a `transitions.txt` template file
   - Ready for you to add timestamp data

3. **Created automation scripts**:
   - `batch_process.py` - Process all videos at once
   - `validate_ground_truth.py` - Compare model vs manual timestamps
   - `rename_and_setup.py` - Script used for initial setup

4. **Updated default config**:
   - Changed to `color_mode: color` (from gray)
   - Added `edge_threshold: 4.0` for better transition detection

5. **Created documentation**:
   - `DATASET.md` - Complete dataset organization guide
   - This file - Quick reference

## 📋 Your Next Steps

### Step 1: Add Transition Timestamps

For each video, add timestamps to its ground truth file:

```bash
# Example: Edit chemistry_01_english
notepad data/ground_truth/chemistry_01_english/transitions.txt
```

**Format:** Two options:

1. **Enhanced** (with ideal frames - recommended):
```
10.5 | 8.3
25.3 | 23.1
45.0 | 44.2
```
Format: `transition | ideal_frame` where ideal_frame is the best moment to capture (1-10s before transition)

2. **Simple** (transitions only):
```
10.5
25.3
45.0
```

### Step 2: Process All Videos

Once you've added timestamps for all videos:

```bash
python batch_process.py --color-mode color --edge-threshold 4.0
```

Or process one video at a time:
```bash
python main.py --video data/raw_videos/chemistry_01_english.mp4 ^
               --output data/processed_chemistry_01_english ^
               --color-mode color --resize 640x360
```

### Step 3: Validate Results

Compare model predictions vs your manual timestamps:

```bash
python validate_ground_truth.py --tolerance 5.0
```

This shows:
- **Recall**: % of your timestamps detected by model
- **Precision**: % of model detections that match your timestamps
- Matched, missed, and false positive transitions

### Step 4: Review and Iterate

Based on validation results:
- If recall is low: Lower `--edge-threshold` (try 3.0-3.5)
- If too many false positives: Raise `--edge-threshold` (try 5.0-6.0)
- Adjust `--tolerance` if timestamps don't align well

## 📊 Current Dataset

**14 PW Lecture Videos:**
- Chemistry: 8 videos (4 Hindi, 4 English)
- Mathematics: 3 videos (2 Hindi, 1 English)
- Physics: 2 videos (1 Hindi, 1 English)  
- Database: 2 videos (2 Hindi)
- Computer Networks: 1 video (1 Hindi)
- Algorithms: 1 video (1 Hindi)

**Test Videos:**
- `input_video_3.mp4`: 100% accuracy (10/10 transitions)
- `input_video_4.mp4`: 83% accuracy (15/18 transitions)

## 🎯 Expected Accuracy

Based on test videos:
- **PPT/Smartboard lectures**: 80-100% accuracy
- **Whiteboard instant-erase**: Not supported (skip these)

## 💡 Tips

1. **Watch at 1.5x or 2x speed** to quickly find transitions
2. **Timestamp when slide is fully visible** (just before next transition)
3. **Ignore micro-transitions** (quick back-and-forth, same slide)
4. **Round to nearest 0.1 second** (e.g., 10.5 instead of 10.47)
5. **Save often** while watching long videos

## 🔧 Troubleshooting

**Video not processing?**
- Check file exists in `data/raw_videos/`
- Check video name matches format in ground truth folder

**Low accuracy in validation?**
- Try lower `--edge-threshold` (default 4.0, try 3.0-3.5)
- Check if ground truth timestamps are accurate
- Increase `--tolerance` in validation (default 5.0, try 7.0-10.0)

**Too many false positives?**
- Raise `--edge-threshold` (try 5.0-6.0)
- Check if video has много animations/movements

## 📁 File Locations

- **Videos**: `data/raw_videos/*.mp4`
- **Ground truth**: `data/ground_truth/<video_id>/transitions.txt`
- **Processed output**: `data/processed_<video_id>/`
- **Validation results**: `validation_results.csv`

## 🚀 After Dataset Collection

Once all videos are processed and validated:
1. Merge metadata with ground truth labels
2. Create balanced training dataset
3. Train ML classifier on labeled frames
4. Integrate trained model into pipeline
5. Deploy for automated slide extraction

---

**Questions?** Check [DATASET.md](DATASET.md) or [WORKFLOW.md](WORKFLOW.md) for detailed guides.
