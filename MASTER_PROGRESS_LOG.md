# Master Progress Log (Full History)

**Project:** Slide Transition Detection System  
**Author:** Project team  
**Last Updated:** January 21, 2026  

---

## 0) Purpose of This File

This document is a single, end-to-end narrative of everything done from scratch to the current state, including all key technical decisions, scripts created, experiments run, best configurations found, problems encountered, and the most recent project organization. It is intended to let anyone pick up the project without missing context.

If you are new to the project, read this file once fully, then follow the “How to Continue” section at the end.

---

## 1) Problem Statement (Initial Goal)

We need to detect **slide transitions** in long lecture videos and extract **clean, high-quality slide images** for downstream use (OCR, indexing, summaries, navigation). The system must work across different lecture styles:

- PowerPoint slides
- Handwritten boards (chalk/whiteboard)
- Mixed content with teachers moving around

The goal is to get **all real transitions** (high recall), then reduce false positives through smart post-processing and best-frame selection.

---

## 2) Initial Pipeline (From Scratch)

### 2.1 Feature Extraction (Per Frame)
We use a 6D feature vector for each frame:

1. `content_fullness`
2. `frame_quality`
3. `is_occluded`
4. `skin_ratio`
5. `edge_change`
6. `frame_diff_mean`

These were extracted using a `FrameFeatureExtractor` inside the core detection script.

### 2.2 Core Model
We used a **GradientBoostingClassifier** (sklearn) trained on labeled data. The model outputs transition probabilities per frame.

### 2.3 Detection Pipeline
After probability prediction, we used a classical post-processing pipeline:

1. **Thresholding** by probability
2. **Smoothing** to reduce noise
3. **Diff-percentile gating** to ensure real changes
4. **Temporal clustering** to group nearby detections
5. **Min-gap merging** to enforce spacing between transitions

---

## 3) Problem Discovery (Low Recall on Non-PPT Styles)

Initial tests showed that the model had **very low recall (near 0%)** on:

- Handwritten board videos
- Dark backgrounds
- Low contrast content

This meant the model was too biased toward PPT-like slides. We needed to improve generalization without losing recall.

---

## 4) Hard Examples Strategy (v3 Training)

To fix recall on difficult videos, we created **hard positive** and **hard negative** samples:

### 4.1 Hard Positives
- Generated frames around real transitions across multiple styles
- File created: `hard_positives.csv`
- Count: ~490 samples

### 4.2 Hard Negatives
- Sampled frames that look like transitions but are not
- File created: `hard_negatives.csv`
- Count: ~1,350 samples

### 4.3 Training v3
We retrained the GradientBoosting model using:

- `labeled_dataset.csv`
- `hard_positives.csv`
- `hard_negatives.csv`

New model:
- `trained_model_gb_enriched_v3.pkl`
- `model_gb_enriched_v3_normalization.pkl`

**Result:** 100% recall across all test videos.

---

## 5) Parameter Sweep (Tight Optimization)

We ran a tight sweep to recover precision while keeping recall.

### 5.1 Script
- `sweep_params.py`

### 5.2 Best Configuration
Found best config:

```
threshold = 0.55
diff_percentile = 90
min_gap = 3.0
smooth_window = 5
```

### 5.3 Results
Best mean F1 ~0.140 (10s tolerance), with 100% recall.

Best results stored in:
- `results_sweep_v3/t0.55_d90_g3.0/`

---

## 6) Post-Filtering for Precision Recovery

Even with optimal parameters, false positives were high. We added a confidence post-filter.

### 6.1 Script
- `detect_with_postfilter.py`

### 6.2 Post-Filter Rule
```
final_threshold = base_threshold + confidence_boost
confidence_boost = 0.10
```

### 6.3 Output
- `results_postfilter_v3_boost010/`

Precision improved slightly but recall stayed at 100%.

---

## 7) Best-Frame Selection (Main Deliverable)

After detecting transitions, we needed the **best slide image** near each transition.

### 7.1 Script
- `select_best_slides.py`

### 7.2 Logic
1. Extract frames around each detected timestamp
2. Score frames using:
   - content_fullness (higher is better)
   - frame_quality (higher is better)
   - is_occluded (lower is better)
   - skin_ratio (lower is better)
   - foreground_ratio (lower is better)
3. Cluster similar frames using aHash (deduplicate)
4. Select best frame per cluster

### 7.3 Output
- `best_frames_v3/`

---

## 8) Teacher Occlusion Problem (New Issue)

User noticed that the best frames often contained a teacher standing in the middle of the slide.

**Goal:** Filter out frames where foreground is centered (teacher blocking the slide).

---

## 9) Edge-Zone Teacher Filter (Attempted)

### 9.1 Idea
If a large foreground blob is centered, drop the frame. If foreground is near edges, keep.

### 9.2 Filter Logic
- Compute foreground mask from frame differences
- Compute `foreground_ratio`
- Compute foreground centroid `center_x`
- Drop frames if:

```
foreground_ratio >= fg_drop
AND center_x in (edge_zone, 1 - edge_zone)
```

### 9.3 Parameters
```
fg_thresh = 0.08
fg_drop = 0.18
edge_zone = 0.20
```

### 9.4 Problem Encountered
Repeated OpenCV memory allocation errors:

- `cv2.error: insufficient memory`
- Even with downscaling and reduced buffers

**Status:** Not fully resolved. Edge-filter outputs are partially complete:
- `best_frames_v3_edge/`

---

## 10) GPU Support (New Capability Added)

Because your friend has a GPU, we added a **PyTorch deep learning model** for GPU acceleration.

### 10.1 New Scripts
- `train_deep_model.py`  (GPU training)
- `detect_gpu.py`        (GPU inference)

### 10.2 Deep Model Architecture
```
6 → 64 → 128 → 64 → 32 → 1
BatchNorm + Dropout + ReLU
```

### 10.3 Expected Benefits
- Training: 2–4 minutes on GPU (vs 10 min CPU)
- Inference: 50–100 FPS (vs 5 FPS CPU)

### 10.4 Dependencies
- `requirements-gpu.txt`
- CUDA + PyTorch

### 10.5 Setup Guide
- `GPU_SETUP_GUIDE.md`

---

## 11) Project Organization and Cleanup

The project had grown large and messy. We reorganized it to make collaboration possible.

### 11.1 Folder Structure
```
models/   → all trained models & datasets
archive/  → old scripts, models, results
docs/     → all documentation
```

### 11.2 .gitignore Updated
We ensured large files and outputs are excluded.

---

## 12) Current Best Model and Results

### 12.1 Best Model
- `models/trained_model_gb_enriched_v3.pkl`
- `models/model_gb_enriched_v3_normalization.pkl`

### 12.2 Best Parameters
```
threshold = 0.55
diff_percentile = 90
min_gap = 3.0
smooth_window = 5
confidence_boost = 0.10
```

### 12.3 Detection Results
- `results_postfilter_v3_boost010/`
- 100% recall on all test videos (10s tolerance)
- Precision ranges 4–13%

### 12.4 Best Frames
- `best_frames_v3/` (initial)
- `best_frames_v3_fg/` (foreground filter)
- `best_frames_v3_edge/` (edge-filter, partial)

---

## 13) Known Issues

1. **Edge-filter memory errors**
   - OpenCV fails to allocate small frames
   - Likely due to memory fragmentation or VideoCapture backend

2. **Low precision**
   - Many false positives
   - By design (recall prioritized)

3. **GPU model not trained yet**
   - Scripts ready
   - Needs execution on GPU laptop

---

## 14) How to Continue (Exact Steps)

### 14.1 On Your Laptop (CPU)

Try to finish edge-filter with smaller window:

```
python select_best_slides.py \
  --videos data/testing_videos \
  --detections results_postfilter_v3_boost010 \
  --output best_frames_final \
  --window 1.0 \
  --step 0.2 \
  --hash-thresh 10 \
  --fg-thresh 0.08 \
  --edge-zone 0.20 \
  --fg-drop 0.18
```

If it still fails, use **post-processing on already saved frames** instead of re-reading video.

### 14.2 On Friend’s GPU Laptop

1. Clone repo
2. Install GPU dependencies (see `GPU_SETUP_GUIDE.md`)
3. Train model:

```
python train_deep_model.py
```

4. Run detection:

```
python detect_gpu.py --video data/testing_videos/algo_1.mp4
```

### 14.3 Compare GPU vs CPU
Use `compare_all_results.py` to compare results and decide which model to adopt.

---

## 15) Files That Matter Most

### Essential Models
- `models/trained_model_gb_enriched_v3.pkl`
- `models/model_gb_enriched_v3_normalization.pkl`
- `models/labeled_dataset.csv`
- `models/hard_positives.csv`
- `models/hard_negatives.csv`

### Core Scripts
- `detect_transitions_universal.py`
- `detect_with_postfilter.py`
- `select_best_slides.py`
- `train_deep_model.py`
- `detect_gpu.py`
- `sweep_params.py`

### Documentation
- `README.md`
- `GPU_SETUP_GUIDE.md`
- `CONTINUATION_GUIDE.md`
- `PROJECT_SUMMARY.md`
- `QUICK_START_CARD.md`

---

## 16) Final Notes

- The **system is functional** and yields full recall.
- The **main remaining task** is to fix edge-filter memory issues and finalize best-frame selection.
- The **GPU path** is strongly recommended for faster iteration and possibly improved accuracy.

---

## 17) Status Summary (Today)

✅ **Completed:**
- v3 training with hard positives/negatives
- Parameter sweep (best config found)
- Post-filtering added
- Best-frame selection implemented
- GPU training + detection scripts created
- Project reorganized and documented

🔄 **In Progress:**
- Edge-filter execution (memory errors)
- GPU model training (not yet run)

❌ **Pending:**
- Final best-frame outputs
- Consolidated CSV of slides
- GPU evaluation and comparison

---

**This file is the single source of truth for project history.**
