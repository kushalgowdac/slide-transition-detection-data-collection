# 🚀 QUICK START CARD

## For YOUR Laptop (CPU)

### Resume Current Work
```bash
# Activate environment
.venv\Scripts\activate

# Run detection
python detect_with_postfilter.py \
    --video data/testing_videos/algo_1.mp4 \
    --model models/trained_model_gb_enriched_v3.pkl \
    --scaler models/model_gb_enriched_v3_normalization.pkl

# Select best frames
python select_best_slides.py \
    --videos data/testing_videos \
    --detections results_postfilter_v3_boost010 \
    --output best_frames_final
```

## For FRIEND's Laptop (GPU) ⚡

### One-Time Setup
```bash
# 1. Check GPU
nvidia-smi

# 2. Clone repo (after you push)
git clone <your-repo-url>
cd "slide transition detection - data collection"

# 3. Setup environment
python -m venv .venv
.venv\Scripts\activate

# 4. Install PyTorch (choose based on nvidia-smi CUDA version)
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5. Install other deps
pip install -r requirements-gpu.txt

# 6. Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
# Should print: GPU: True
```

### Train & Use
```bash
# Train model (~2 minutes on RTX 3060)
python train_deep_model.py

# Detect transitions (10-20x faster!)
python detect_gpu.py \
    --video data/testing_videos/algo_1.mp4 \
    --model models/trained_model_deep.pth
```

## 📚 Documentation

- **Main Guide:** README.md
- **GPU Setup:** GPU_SETUP_GUIDE.md  
- **Continue Work:** CONTINUATION_GUIDE.md
- **Today's Work:** PROJECT_SUMMARY.md

## 📁 Important Files

### Models (models/)
- `trained_model_gb_enriched_v3.pkl` - Current best (CPU)
- `labeled_dataset.csv` - Training data
- `hard_positives.csv` - Hard examples
- `hard_negatives.csv` - Hard examples

### Results
- `results_postfilter_v3_boost010/` - Latest detections
- `best_frames_v3/` - Best slide frames

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Memory error | Reduce `--window 1.0` or `--batch-size 16` |
| CUDA not found | Update NVIDIA drivers, reinstall PyTorch |
| Import error | `pip install -r requirements-gpu.txt` |

## 🎯 Current Best Config

```python
threshold = 0.55
diff_percentile = 90
min_gap = 3.0
smooth_window = 5
confidence_boost = 0.10
```

## 📊 Performance

| Task | CPU | GPU |
|------|-----|-----|
| Training | 30s | 2min |
| Detection | 5 FPS | 50-100 FPS |
| 1hr video | ~40min | ~2min |

---

**Status:** ✅ Ready for GPU training  
**Version:** 3.0  
**Last Updated:** Jan 21, 2026
