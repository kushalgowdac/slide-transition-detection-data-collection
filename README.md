# Slide Transition Detection System

Automated detection of slide transitions in educational lecture videos using machine learning. Supports both CPU-based (sklearn) and **GPU-accelerated (PyTorch)** models.

## 🎯 Overview

This system automatically identifies when slides change in lecture videos, making it easier to:
- Extract key slides from long lectures
- Create video timestamps for slide changes  
- Generate slide decks from recorded lectures
- Improve video navigation and indexing

**Current Performance (v3 Model):**
- **Recall:** 100% (detects all actual transitions)
- **Precision:** 5-13% with post-filtering
- **F1 Score:** ~0.14 (optimized for recall to catch all slides)

## 🚀 Quick Start

### Option 1: CPU-based (Sklearn) - Your Current Laptop
```bash
# Install dependencies
pip install -r requirements.txt

# Detect transitions
python detect_with_postfilter.py \
    --video data/testing_videos/algo_1.mp4 \
    --model models/trained_model_gb_enriched_v3.pkl \
    --scaler models/model_gb_enriched_v3_normalization.pkl \
    --output results_cpu
```

### Option 2: GPU-accelerated (PyTorch) - For Friend's GPU Laptop ⚡
```bash
# Install GPU dependencies (see GPU_SETUP_GUIDE.md)
pip install -r requirements-gpu.txt

# Train deep learning model
python train_deep_model.py \
    --dataset models/labeled_dataset.csv \
    --extra-positives models/hard_positives.csv \
    --extra-negatives models/hard_negatives.csv \
    --epochs 100

# Detect with GPU (10-20x faster!)
python detect_gpu.py \
    --video data/testing_videos/algo_1.mp4 \
    --model models/trained_model_deep.pth \
    --output results_gpu
```

## 📁 Organized Project Structure

```
├── models/                          # 🎯 Trained models (KEEP THESE!)
│   ├── trained_model_gb_enriched_v3.pkl       # Best sklearn model
│   ├── model_gb_enriched_v3_normalization.pkl # Feature scaler
│   ├── labeled_dataset.csv          # Training data (2,851 samples)
│   ├── hard_positives.csv           # Hard examples (490 samples)
│   └── hard_negatives.csv           # Hard examples (1,350 samples)
│
├── 🚀 Core Scripts (Use These!)
│   ├── train_classifier_gb_enriched_v2.py  # Train sklearn model (CPU)
│   ├── train_deep_model.py         # ⚡ Train PyTorch model (GPU)
│   ├── detect_transitions_universal.py     # Universal detection engine
│   ├── detect_with_postfilter.py   # CPU detection + confidence filter
│   ├── detect_gpu.py                # ⚡ GPU-accelerated detection
│   ├── select_best_slides.py       # Best frame selector
│   ├── sweep_params.py              # Find optimal parameters
│   └── compare_with_ground_truth.py # Evaluation
│
├── 🛠️ Utilities
│   ├── generate_hard_positives.py
│   ├── generate_hard_negatives.py
│   ├── compare_all_results.py
│   └── batch_process.py
│
├── 📊 Current Results (Continue from here!)
│   ├── results_postfilter_v3_boost010/  # Latest detections
│   ├── results_sweep_v3/           # Parameter sweep
│   ├── results_enriched_v3_best/   # Best config results
│   ├── best_frames_v3/             # Best slide frames
│   ├── best_frames_v3_fg/          # With foreground filter
│   └── best_frames_v3_edge/        # With edge-zone filter (incomplete)
│
├── data/
│   ├── testing_videos/             # Test videos
│   ├── ground_truth/               # Ground truth timestamps
│   └── annotations/                # Feature annotations
│
├── docs/                           # 📚 Documentation (reference)
├── archive/                        # 🗑️ Old files (can ignore)
│   ├── old_models/                 # Previous model versions
│   ├── old_results/                # Previous results
│   ├── old_scripts/                # Deprecated scripts
│   └── old_docs/                   # Old documentation
│
└── ⚙️ Configuration
    ├── requirements.txt            # CPU dependencies
    ├── requirements-gpu.txt        # ⚡ GPU dependencies
    ├── GPU_SETUP_GUIDE.md          # GPU setup instructions
    └── .gitignore                  # Updated for organization
```

## 🔧 Installation

### For Your Laptop (CPU)
```bash
# Clone repository
git clone <your-repo-url>
cd "slide transition detection - data collection"

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify
python -c "import sklearn, cv2; print('✓ Ready!')"
```

### For Friend's GPU Laptop ⚡
```bash
# Clone repository  
git clone <your-repo-url>
cd "slide transition detection - data collection"

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Check CUDA version first!
nvidia-smi  # Look for "CUDA Version: XX.X"

# Install PyTorch with matching CUDA
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1+:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements-gpu.txt

# Verify GPU setup
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"
```

**Expected GPU output:**
```
GPU Available: True
Device: NVIDIA GeForce RTX 3060
```

See [GPU_SETUP_GUIDE.md](GPU_SETUP_GUIDE.md) for detailed instructions and troubleshooting.

## 🎓 Training Models

### Sklearn Model (CPU) - Current Best
```bash
python train_classifier_gb_enriched_v2.py \
    --dataset models/labeled_dataset.csv \
    --extra-positives models/hard_positives.csv \
    --extra-negatives models/hard_negatives.csv \
    --output models/
```
**Time:** ~30 seconds

### Deep Learning Model (GPU) - For Better Performance ⚡
```bash
python train_deep_model.py \
    --dataset models/labeled_dataset.csv \
    --extra-positives models/hard_positives.csv \
    --extra-negatives models/hard_negatives.csv \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --dropout 0.3 \
    --output models/
```
**Time:** ~2 minutes (GPU), ~10 minutes (CPU)

## 🔍 Complete Workflow

### 1. Parameter Optimization
Find best parameters for your videos:
```bash
python sweep_params.py \
    --videos data/testing_videos \
    --ground-truth data/ground_truth \
    --model models/trained_model_gb_enriched_v3.pkl \
    --output results_sweep/
```

### 2. Detection
Run detection with optimized parameters:
```bash
# CPU
python detect_with_postfilter.py \
    --video data/testing_videos/algo_1.mp4 \
    --model models/trained_model_gb_enriched_v3.pkl \
    --threshold 0.55 \
    --diff-pct 90 \
    --min-gap 3.0 \
    --confidence-boost 0.10 \
    --output results/

# GPU (10-20x faster!)
python detect_gpu.py \
    --video data/testing_videos/algo_1.mp4 \
    --model models/trained_model_deep.pth \
    --threshold 0.55 \
    --output results_gpu/
```

### 3. Best Frame Selection
Extract best slide frames:
```bash
python select_best_slides.py \
    --videos data/testing_videos \
    --detections results_postfilter_v3_boost010 \
    --output best_frames/ \
    --window 2.0 \
    --hash-thresh 10 \
    --fg-thresh 0.08 \
    --edge-zone 0.20 \
    --fg-drop 0.18
```

### 4. Evaluation
```bash
python compare_with_ground_truth.py \
    --detected results/algo_1_detected.txt \
    --ground-truth data/ground_truth/algo_1_gt.txt \
    --tolerance 10.0
```

## 📊 Current Best Configuration

```python
# Optimized parameters (from parameter sweep)
threshold = 0.55          # Probability threshold
diff_percentile = 90      # Frame difference percentile  
min_gap = 3.0            # Minimum seconds between transitions
smooth_window = 5        # Smoothing window size
confidence_boost = 0.10  # Post-filter boost
```

**Performance (10s tolerance):**
| Video | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| algo_1 | 12.8% | 100% | 0.227 |
| cn_1 | 5.5% | 100% | 0.104 |
| db_1 | 5.7% | 100% | 0.107 |
| toc_1 | 4.4% | 100% | 0.084 |

## ⚡ GPU vs CPU Comparison

| Metric | CPU (Sklearn) | GPU (PyTorch) |
|--------|--------------|---------------|
| **Training Time** | 30s | 2min (GPU) / 10min (CPU) |
| **Inference Speed** | ~5 FPS | ~50-100 FPS |
| **Model Size** | 200KB | 500KB |
| **Accuracy** | F1: 0.807 (test) | Similar or better expected |
| **Memory Usage** | 500MB | 2GB (GPU) / 1GB (CPU) |
| **Best For** | Quick tests, no GPU | Batch processing, large videos |

**💡 Recommendation:**
- **Your laptop:** Use CPU (sklearn) - already trained and working
- **Friend's laptop:** Train PyTorch model on GPU for 10-20x faster processing

## 🐛 Troubleshooting

### Memory Errors (Current Issue)
```
cv2.error: insufficient memory
```
**Solutions:**
1. Reduce batch size: `--batch-size 16`
2. Process smaller video chunks
3. Use GPU (more memory available)

### CUDA Not Found
```
RuntimeError: CUDA not available
```
**Check:**
```bash
nvidia-smi  # Verify GPU and CUDA version
python -c "import torch; print(torch.cuda.is_available())"
```

See [GPU_SETUP_GUIDE.md](GPU_SETUP_GUIDE.md) for complete troubleshooting guide.

## 📈 Key Information to Continue Work

### Essential Files (Already in models/)
- `trained_model_gb_enriched_v3.pkl` - Current best model (F1: 0.807)
- `model_gb_enriched_v3_normalization.pkl` - Feature scaler
- `labeled_dataset.csv` - Base training data (2,851 samples)
- `hard_positives.csv` - 490 positive samples from all test videos
- `hard_negatives.csv` - 1,350 negative samples

### Latest Results (Continue from here!)
- `results_postfilter_v3_boost010/` - Detection with confidence filter
- `best_frames_v3/` - Best slide frames (includes some teacher-blocking)
- `best_frames_v3_fg/` - With foreground filtering
- `best_frames_v3_edge/` - With edge-zone filtering (incomplete - memory errors)

### Current State
✅ **Completed:**
- v3 model trained with hard positives/negatives
- Parameter sweep done (best: thresh=0.55, diff-pct=90, min-gap=3.0)
- 100% recall achieved on all test videos
- Best-frame selection implemented

🔄 **In Progress:**
- Edge-zone teacher filter (memory allocation errors)
- GPU model training

❌ **Pending:**
- Resolve memory errors
- Train PyTorch model on GPU
- Final best-frame selection
- Consolidated output CSV

### How to Resume
```bash
# 1. For CPU work (current approach)
python select_best_slides.py \
    --videos data/testing_videos \
    --detections results_postfilter_v3_boost010 \
    --output best_frames_final \
    --window 1.0  # Reduced window to save memory

# 2. For GPU work (recommended for friend's laptop)
# First, train GPU model
python train_deep_model.py --epochs 100

# Then use for detection
python detect_gpu.py \
    --video data/testing_videos/algo_1.mp4 \
    --model models/trained_model_deep.pth
```

## 🎯 Next Steps

### Immediate (Current System)
1. Fix memory errors in edge-zone filter
2. Complete best-frame selection
3. Create consolidated CSV output

### For GPU Laptop
1. Install PyTorch with CUDA (see GPU_SETUP_GUIDE.md)
2. Train deep learning model
3. Compare performance vs sklearn
4. Use for faster batch processing

### Future Improvements
- Add temporal context (LSTM/GRU)
- Transfer learning with pre-trained models
- OCR-based text change detection
- Better post-processing for precision

## 📚 Documentation

- **[GPU_SETUP_GUIDE.md](GPU_SETUP_GUIDE.md)** - Detailed GPU setup for friend's laptop
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development guidelines
- **docs/** - Additional technical documentation and old files

## 📝 Citation

If you use this system, please cite:
```bibtex
@software{slide_transition_detection,
  title={Slide Transition Detection System},
  author={Your Name},
  year={2026},
  description={Automated slide transition detection in lecture videos using ML}
}
```

---

**Status:** Active Development  
**Version:** 3.0 (GPU Support Added)  
**Last Updated:** January 21, 2026

**Quick Links:**
- 🐛 [Issues](issues/) - Report bugs or request features
- 💬 [Discussions](discussions/) - Ask questions
- 📖 [Wiki](wiki/) - Additional documentation
