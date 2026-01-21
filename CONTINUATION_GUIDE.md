# Continuation Guide - Quick Reference

Essential information to continue working on the project.

## 📍 Current State (Where We Left Off)

### ✅ Completed Work
1. **Model v3 Training**
   - Trained with hard positives/negatives
   - Features: 6D vector (content, quality, occlusion, skin, edge, diff)
   - Performance: F1=0.807 on test set, 100% recall on videos

2. **Parameter Optimization**
   - Ran 12-config sweep
   - Best config: threshold=0.55, diff-pct=90, min-gap=3.0
   - Results in `results_sweep_v3/`

3. **Detection Pipeline**
   - Detection with best config: `results_enriched_v3_best/`
   - Post-filtering: `results_postfilter_v3_boost010/`
   - 100% recall, 4-13% precision (optimized for recall)

4. **Best Frame Selection**
   - Initial: `best_frames_v3/`
   - With foreground filter: `best_frames_v3_fg/`
   - Partial edge-filter: `best_frames_v3_edge/` (incomplete)

5. **Project Cleanup**
   - Organized files into `models/`, `docs/`, `archive/`
   - Updated `.gitignore`
   - Created comprehensive documentation

6. **GPU Support Added**
   - `train_deep_model.py` - PyTorch deep learning model
   - `detect_gpu.py` - GPU-accelerated detection
   - `requirements-gpu.txt` - GPU dependencies
   - `GPU_SETUP_GUIDE.md` - Complete setup guide

### 🔄 In Progress (Unfinished)
1. **Edge-zone teacher filter** - Memory allocation errors during execution
2. **GPU model training** - Script ready, not yet trained

### ❌ Pending Tasks
1. Resolve memory errors in edge-filter
2. Train PyTorch model on GPU
3. Complete final best-frame selection
4. Create consolidated CSV output
5. GitHub repository update

## 📂 Essential Files (DO NOT DELETE)

### Models (in models/)
- `trained_model_gb_enriched_v3.pkl` - Best sklearn model
- `model_gb_enriched_v3_normalization.pkl` - Feature scaler
- `labeled_dataset.csv` - 2,851 training samples
- `hard_positives.csv` - 490 hard positive examples
- `hard_negatives.csv` - 1,350 hard negative examples

### Core Scripts
- `train_classifier_gb_enriched_v2.py` - Train sklearn model
- `train_deep_model.py` - Train PyTorch model (GPU)
- `detect_transitions_universal.py` - Core detection engine
- `detect_with_postfilter.py` - CPU detection
- `detect_gpu.py` - GPU detection
- `select_best_slides.py` - Best frame selector
- `sweep_params.py` - Parameter optimization
- `compare_with_ground_truth.py` - Evaluation
- `generate_hard_positives.py` - Generate hard examples
- `generate_hard_negatives.py`

### Current Results (Continue from here!)
- `results_postfilter_v3_boost010/` - Latest detections
- `results_enriched_v3_best/` - Detection with best config
- `results_sweep_v3/` - Parameter sweep results
- `best_frames_v3/` - Best frames (has teacher-blocking)
- `best_frames_v3_fg/` - With foreground filter
- `best_frames_v3_edge/` - With edge filter (incomplete)

## 🚀 How to Continue

### Option 1: Resume Current Work (CPU)

**Fix memory errors and complete edge-filter:**
```bash
# Try with smaller window
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

**Alternative approach - post-process existing frames:**
1. Load already-saved frames from `best_frames_v3/`
2. Apply edge-filter as post-processing step
3. Avoids re-reading videos (saves memory)

### Option 2: GPU Workflow (Friend's Laptop)

**1. Setup (once):**
```bash
# Follow GPU_SETUP_GUIDE.md
pip install -r requirements-gpu.txt
python -c "import torch; print(torch.cuda.is_available())"
```

**2. Train GPU model:**
```bash
python train_deep_model.py \
    --dataset models/labeled_dataset.csv \
    --extra-positives models/hard_positives.csv \
    --extra-negatives models/hard_negatives.csv \
    --epochs 100 \
    --batch-size 32
```

**3. Run detection:**
```bash
python detect_gpu.py \
    --video data/testing_videos/algo_1.mp4 \
    --model models/trained_model_deep.pth \
    --threshold 0.55 \
    --output results_gpu
```

**4. Compare with sklearn:**
```bash
python compare_all_results.py \
    --results-dirs results_postfilter_v3_boost010,results_gpu \
    --ground-truth data/ground_truth \
    --output comparison.csv
```

## 🔧 Current Best Parameters

From parameter sweep (`results_sweep_v3/`):
```python
threshold = 0.55          # Probability threshold
diff_percentile = 90      # Frame difference percentile
min_gap = 3.0            # Minimum gap between transitions (seconds)
smooth_window = 5        # Smoothing window
confidence_boost = 0.10  # Post-filter confidence boost
```

## 📊 Current Performance

### Detection (10s tolerance)
| Video | Transitions | Precision | Recall | F1 |
|-------|------------|-----------|--------|-----|
| algo_1 | 87 | 12.8% | 100% | 0.227 |
| cn_1 | 172 | 5.5% | 100% | 0.104 |
| db_1 | 159 | 5.7% | 100% | 0.107 |
| toc_1 | 172 | 4.4% | 100% | 0.084 |

### Model (Test Set)
- Precision: 0.757
- Recall: 0.864
- F1 Score: 0.807
- AUC: ~0.95

## 🐛 Known Issues

### 1. Memory Allocation Errors
**Symptom:** `cv2.error: insufficient memory` during edge-filter  
**Status:** Unresolved  
**Workaround:** Reduce window size, process one video at a time

### 2. Low Precision
**Symptom:** 88-96% of detections are false positives  
**Status:** Expected (optimized for recall)  
**Next:** Try GPU model with deeper architecture

### 3. Edge-filter Incomplete
**Symptom:** `best_frames_v3_edge/` only partially complete  
**Status:** Interrupted due to memory errors  
**Next:** Try alternative implementation or GPU processing

## 📝 Next Immediate Steps

### Priority 1: Complete Current Work
1. [ ] Fix memory errors in edge-filter
2. [ ] Generate final best-frame outputs
3. [ ] Create consolidated CSV with all best slides
4. [ ] Visual inspection of results

### Priority 2: GPU Training
1. [ ] Set up friend's GPU laptop (follow GPU_SETUP_GUIDE.md)
2. [ ] Train PyTorch model
3. [ ] Compare performance vs sklearn
4. [ ] Document results

### Priority 3: Improvements
1. [ ] Add temporal context (LSTM)
2. [ ] Experiment with transfer learning
3. [ ] Implement stricter post-processing
4. [ ] Try different architectures

## 🎯 Quick Commands Reference

```bash
# Current best detection (CPU)
python detect_with_postfilter.py \
    --video VIDEO.mp4 \
    --model models/trained_model_gb_enriched_v3.pkl \
    --threshold 0.55 --diff-pct 90 --min-gap 3.0 \
    --confidence-boost 0.10

# GPU detection (after training)
python detect_gpu.py \
    --video VIDEO.mp4 \
    --model models/trained_model_deep.pth

# Best frame selection
python select_best_slides.py \
    --videos data/testing_videos \
    --detections results_postfilter_v3_boost010 \
    --output best_frames/

# Parameter sweep
python sweep_params.py \
    --videos data/testing_videos \
    --ground-truth data/ground_truth \
    --model models/trained_model_gb_enriched_v3.pkl

# Evaluation
python compare_with_ground_truth.py \
    --detected results/video_detected.txt \
    --ground-truth data/ground_truth/video_gt.txt
```

## 📚 Documentation Map

- **README.md** - Main overview and quick start
- **GPU_SETUP_GUIDE.md** - Complete GPU setup guide
- **CONTINUATION_GUIDE.md** - This file
- **docs/** - Additional documentation and old files
- **archive/** - Old scripts, models, results (reference only)

## 🔗 GitHub Repository Setup

### Before Pushing
1. [ ] Verify `.gitignore` excludes large files
2. [ ] Check all essential files are tracked
3. [ ] Write meaningful commit message
4. [ ] Tag important versions

### Recommended Structure
```
main branch:
├── Clean organized code
├── Essential documentation
├── Current best model
└── Latest results (summary only)

gh-pages (optional):
└── Project website/documentation

releases:
└── v3.0 - GPU support added
```

### Commit Message Example
```
feat: Add GPU support and organize project structure

- Add PyTorch deep learning model (train_deep_model.py)
- Add GPU-accelerated detection (detect_gpu.py)
- Organize files into models/, docs/, archive/
- Create comprehensive GPU setup guide
- Update README with GPU instructions
- Clean up workspace (move old files to archive)

Current state:
- v3 model: F1=0.807, 100% recall
- Best config: thresh=0.55, diff-pct=90, min-gap=3.0
- GPU support ready for training

Known issues:
- Edge-filter memory errors (in progress)
- Low precision (expected, optimized for recall)
```

## 💡 Tips for Friend's GPU Setup

1. **Before cloning:** Ensure NVIDIA drivers are up to date
2. **After cloning:** Run GPU verification immediately
3. **Training:** Start with small epochs (10) to test, then full 100
4. **Monitoring:** Keep `nvidia-smi -l 1` running in separate terminal
5. **Comparison:** Save both CPU and GPU results for comparison

## ⚠️ Important Notes

1. **Don't delete archive/** - Contains history and reference files
2. **Keep models/** synced - Both need same dataset files
3. **Back up results/** - Detection results are valuable
4. **Document changes** - Update this guide as you progress
5. **Test before committing** - Run quick tests on sample videos

## 📞 Quick Troubleshooting

| Issue | Quick Fix |
|-------|-----------|
| Memory error | Reduce batch size or window size |
| CUDA not available | Check `nvidia-smi`, reinstall PyTorch |
| Import error | `pip install -r requirements-gpu.txt` |
| Slow training | Increase batch size (if memory allows) |
| Low accuracy | Normal - optimized for recall (100%) |

---

**Last Updated:** January 21, 2026  
**Current Version:** v3.0 (GPU Support)  
**Status:** Ready for GPU training and final frame selection
