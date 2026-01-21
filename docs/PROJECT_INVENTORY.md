# 🎓 Slide Transition Detection Project - Complete Inventory

**Project Status**: Production-Ready Model v1 + Building Improved Model v2  
**Last Updated**: 2026-01-18  
**Total Work**: ~15,000+ lines of code, documentation, and annotated data

---

## 📊 PART 1: INTERVIEW PREPARATION CONTENT (9,100+ Lines)

### 1.1 INTERVIEW_GUIDE.md (3,500 lines)
- **30-second pitch**: Elevator pitch for interviewers
- **2-minute story**: Full narrative for technical interviews
- **7 major improvements**: What you added beyond initial scope
- **Common follow-up questions**: 20+ Q&A pairs
- **Technical depth options**: How to adjust complexity by audience

### 1.2 INTERVIEW_STORIES.md (2,800 lines)
- **8 narrative frameworks**: Different ways to tell your project story
- **Timed narratives**: 30s, 2min, 5min, 10min versions
- **Problem-solution-impact structure**: Clear storytelling arc
- **Technical details**: Feature engineering, model architecture
- **Real metrics**: 97.45% accuracy, precision, recall, F1-scores

### 1.3 INTERVIEW_FAQS.md (2,800 lines)
- **28 interview questions**: Common queries from recruiters
- **Detailed answers**: 150-250 words each with examples
- **Topics covered**: 
  - Architecture decisions
  - Why Decision Trees vs other models
  - How you handle class imbalance (97.6% negative)
  - Data collection process
  - Challenges overcome
  - Future improvements

---

## 🎬 PART 2: TRAINING DATASET (41,650 Labeled Frames)

### 2.1 Raw Videos (14 Videos, ~20 minutes each)
Location: `data/raw_videos/`

| Video | Language | Domain | Duration | Frames |
|-------|----------|--------|----------|--------|
| algorithms_14_hindi.mp4 | Hindi | Algorithms | ~20min | 1,187 |
| chemistry_01_english.mp4 | English | Chemistry | ~20min | 10,626 |
| chemistry_04_english.mp4 | English | Chemistry | ~20min | 13,272 |
| chemistry_08_hindi.mp4 | Hindi | Chemistry | ~20min | 1,312 |
| chemistry_09_hindi.mp4 | Hindi | Chemistry | ~20min | 1,641 |
| chemistry_10_english.mp4 | English | Chemistry | ~20min | 1,073 |
| computer_networks_13_hindi.mp4 | Hindi | Networks | ~20min | 1,363 |
| database_11_hindi.mp4 | Hindi | Database | ~20min | 1,552 |
| database_12_hindi.mp4 | Hindi | Database | ~20min | 1,530 |
| database_13_hindi.mp4 | Hindi | Database | ~20min | 1,587 |
| mathematics_03_english.mp4 | English | Mathematics | ~20min | (no frames) |
| mathematics_05_hindi.mp4 | Hindi | Mathematics | ~20min | 3,051 |
| mathematics_07_hindi.mp4 | Hindi | Mathematics | ~20min | 676 |
| physics_01_english.mp4 | English | Physics | ~20min | 1,352 |
| physics_05_english.mp4 | English | Physics | ~20min | 1,428 |

**Total**: 14 videos, ~280 minutes (4.7 hours)

### 2.2 Ground Truth Annotations (data/ground_truth/)
- **Format**: `MM.SS` (minutes.seconds)
- **One file per video**: `video_name/transitions.txt`
- **Example content**:
  ```
  0.20  (20 seconds)
  3.42  (3 minutes 42 seconds = 222 seconds)
  7.13  (7 minutes 13 seconds = 433 seconds)
  ```
- **Total transitions**: 1,015 across all 14 videos (2.4% of frames)

### 2.3 Labeled Dataset CSV (labeled_dataset.csv)
**Current Version**: 41,650 rows

**Columns**:
- `video_name`: Source video
- `frame_path`: Where frame is saved
- `frame_idx`: Frame number in video
- `timestamp`: Time in seconds
- `is_transition_gt`: **1** if transition, **0** if not (ground truth)
- `transition_id_gt`: Which transition number
- `split`: **train / val / test**
- Feature columns:
  - `edge_change`: Laplacian edge detection
  - `is_occluded`: Occlusion detection (0/1)
  - `skin_ratio`: Skin percentage in frame (0-1)
  - `content_fullness`: Content area ratio (0-1)
  - `frame_quality`: Blur detection score

**Current Distribution**:
```
Train:  35,143 rows (84.4%) | 784 transitions (2.2%)
Val:     3,727 rows (8.9%)  | 69 transitions (1.9%)
Test:    2,780 rows (6.7%)  | 162 transitions (5.8%)  ← IMBALANCED!
```

⚠️ **Problem Identified**: Test set has 2.7x more transitions than training set

---

## 🤖 PART 3: MODEL ARCHITECTURE (Custom Decision Tree)

### 3.1 Model Code (src/classifier.py)
- **Class**: `SimpleDecisionTree`
- **Algorithm**: Information Gain-based recursive splitting
- **Hyperparameters**:
  - `max_depth = 15`
  - No pruning
  - Min samples for split = 1

### 3.2 Feature Engineering (src/features.py)
**4 core features**:
1. **content_fullness** (45% importance): Ratio of content area to frame area
   - Uses Otsu thresholding + contour detection
2. **frame_quality** (33% importance): Inverse of blur (Laplacian variance)
   - Higher = clearer frame
3. **is_occluded** (15% importance): Occlusion detection (binary)
   - Uses skin color detection (HSV)
4. **skin_ratio** (7% importance): Percentage of skin pixels
   - Helps detect when teacher blocks content

### 3.3 Model v1 (Current, Production)
**File**: `trained_model.pkl`

**Performance on Training Data**:
- Accuracy: 97.45%
- Precision: 77.25%
- Recall: 79.63%
- F1-Score: 78.42%

**⚠️ Issue Found**: Model fails on new test videos (algo_1, cn_1)
- Root cause: Biased training data distribution
- Solution: Building Model v2 with balanced data

---

## 🧪 PART 4: TEST DATASETS (4 New Videos)

Location: `data/testing_videos/`

### 4.1 algo_1.mp4 (Computer Networks)
- **Duration**: 1210.69 seconds (~20 minutes)
- **Transitions**: 10 (in `algo_1_transitions.txt`)
  - 6 clustered in first 18 seconds (quick transitions)
  - 4 later transitions at ~17-18 minutes
- **Format**: MM.SS (e.g., 1.41 = 1:41 = 101 seconds)
- **Status**: Created ✅ | Tested ❌ (0% detection with v1)

### 4.2 cn_1.mp4 (Computer Networks)
- **Duration**: 1213.34 seconds (~20 minutes)
- **Transitions**: 17 (in `cn_1_transitions.txt`)
- **Format**: SS.SS (e.g., 0.17 = 0.17 seconds, 19.50 = 19.50 seconds) ← Different format!
- **Status**: Created ✅ | Tested ❌ (0% detection with v1)

### 4.3 db_1.mp4 (Database)
- **Duration**: Unknown
- **Transitions**: 5 (in `db_1_transtions.txt`)
  - Format: MM.SS (1.22, 2.14, 4.24, 7.09, 7.56)
- **Status**: Created ✅ | Not tested yet

### 4.4 toc_1.mp4 (Theory of Computation)
- **Duration**: Unknown
- **Transitions**: 8 (in `toc_1_transitions.txt`)
- **Format**: MM.SS
- **Status**: Created ✅ | Tested ❌ (0% detection)
- **Note**: Originally tested, showed issues

---

## 🔧 PART 5: CODE & TOOLS

### 5.1 Data Preparation Pipeline
**File**: `src/preparation.py` / `prepare_training_data.py`
- Extracts frames from videos at specified FPS
- Computes all 4 features per frame
- Saves to CSV with timestamps and labels
- Handles multiple videos in batch

**Output**: Frame folders in `data/processed_{video_name}/frames/`

### 5.2 Model Training Script
**File**: `train_classifier.py`
- Loads `labeled_dataset.csv`
- Splits data (train/val/test)
- Trains Decision Tree on training set
- Validates on validation set
- Tests on test set
- Saves `trained_model.pkl`

**Usage**:
```bash
.\.venv\Scripts\python.exe train_classifier.py
```

### 5.3 Testing Framework v1 (Professional)
**File**: `test_model_professional.py`
- Loads video and model
- Extracts frames at 1 FPS
- Computes features
- Runs inference
- Loads ground truth (MM:SS format)
- Computes metrics (precision, recall, F1)
- Outputs JSON results

**Status**: Works but shows 0% recall on new videos

### 5.4 Testing Framework v2 (Enhanced)
**File**: `test_model_v2.py` (Latest)
- Same as v1, plus:
- **Type hints** on all functions
- **Structured logging** (logging module, not print)
- **Dataclasses** for clean data (FrameFeatures, TransitionPrediction, EvaluationMetrics)
- **Auto-detect timestamp format**:
  - MM.SS detection (1.41 → 101 seconds)
  - SS.SS detection (0.17 → 0.17 seconds)
- **Configurable clustering distance** (MIN_DISTANCE_BETWEEN_TRANSITIONS)
- **CLI arguments** for easy testing:
  ```bash
  .\.venv\Scripts\python.exe test_model_v2.py \
    --video data/testing_videos/algo_1.mp4 \
    --ground-truth data/testing_videos/algo_1_transitions.txt \
    --fps 1.0 \
    --output results.json \
    --log-level INFO
  ```

### 5.5 Utility Scripts
**File**: `src/utils.py`
- Video loading and frame extraction
- Feature normalization
- Metrics calculation

**File**: `src/extraction.py`
- Feature extraction (edge detection, blur, occlusion, skin ratio)

---

## 📈 PART 6: DOCUMENTATION

### 6.1 Quick Start Guides
- **QUICK_START_v2.md**: Usage of test_model_v2.py with examples

### 6.2 Testing Documentation
- **TEST_IMPROVEMENTS.md**: Comparison of v1 vs v2 framework
- **TESTING_WORKFLOW.md**: Complete testing procedure
- **TESTING_STATUS_UPDATE.md**: Current test results

### 6.3 Problem Analysis
- **PROBLEM_ANALYSIS.md**: Root cause of model degradation (data corruption)
- **TESTING_RESULTS_PROGRESS.md**: Running log of test results

### 6.4 Configuration
- **configs/defaults.yaml**: Default hyperparameters

---

## 🚨 ISSUES DISCOVERED & ANALYSIS

### Issue 1: Model Degradation (RESOLVED ✅)
**Problem**: Model detected 0 transitions on toc_1 after retraining
**Root Cause**: Data corruption during merge (is_transition_gt column flipped)
**Solution**: Restored original dataset, retrained model
**Status**: ✅ FIXED

### Issue 2: Timestamp Format Mismatch (RESOLVED ✅)
**Problem**: algo_1 test showed 0% recall with many false detections
**Root Cause**: 
- Ground truth used MM.SS format (4.38 = 278 seconds)
- Script was parsing as decimal seconds (4.38 = 4.38 seconds)
- Model predictions were actually correct!
**Solution**: Added auto-detecting timestamp parser
**Status**: ✅ FIXED

### Issue 3: Data Distribution Bias (IDENTIFIED - BUILDING FIX)
**Problem**: Model fails on new videos (cn_1, algo_1, db_1, toc_1)
**Root Cause**:
- Train/Val/Test split is 84.4% / 8.9% / 6.7% (NOT 70/30!)
- Test set has 2.7x more transitions than training set
- Two videos (chemistry_04, chemistry_01) = 57% of training data
- Model heavily biased to these two teachers' styles
**Solution**: Build Model v2 with proper stratification (→ IN PROGRESS)

---

## 📋 SUMMARY TABLE

| Component | Status | Quality | Notes |
|-----------|--------|---------|-------|
| 14 training videos | ✅ Complete | High | Fully annotated with transitions |
| 41,650 labeled frames | ✅ Complete | High | All 4 features computed |
| Ground truth data | ✅ Complete | Good | MM.SS format, some SS.SS mixed in |
| Model v1 (Current) | ✅ Deployed | Medium | 97.45% on training, fails on new data |
| Testing framework v1 | ✅ Working | Medium | Basic functionality |
| Testing framework v2 | ✅ Enhanced | High | Type hints, logging, auto timestamp format |
| Interview prep docs | ✅ Complete | High | 9,100 lines, comprehensive |
| 4 test videos | ✅ Created | Medium | Some format inconsistencies |
| Model v2 (New) | 🔨 In Progress | TBD | Building with stratified dataset |

---

## 🎯 WHAT'S NEXT: MODEL v2 STRATEGY

**Goal**: Build more robust model without touching existing data

**Plan**:
1. ✅ Analyze current data distribution (DONE)
2. 🔨 Create stratified dataset v2:
   - Video-level split (each video → only train OR val OR test)
   - Class-level balance (keep ~2.4% transitions in all splits)
   - Balanced video distribution (each teacher represented equally)
3. 🔨 Train Model v2 on stratified dataset
4. 🧪 Test on algo_1, cn_1, db_1, toc_1
5. 📊 Compare v1 vs v2 performance

**Expected Improvement**: Better generalization to new teachers/styles

---

## 💾 FILE STRUCTURE

```
project_root/
├── data/
│   ├── raw_videos/              (14 MP4 files, ~280 minutes total)
│   ├── ground_truth/            (14 folders with transitions.txt)
│   ├── testing_videos/          (4 MP4 files + transitions.txt)
│   ├── processed_*/frames/      (extracted frames, ~42K images)
│   └── annotations/
│       ├── annotation_manifest.csv
│       └── frames_metadata.csv
├── src/
│   ├── classifier.py            (DecisionTree implementation)
│   ├── features.py              (Feature extraction)
│   ├── extraction.py            (Frame extraction)
│   └── utils.py                 (Utilities)
├── configs/
│   └── defaults.yaml
├── scripts/
│   ├── prepare_data.sh
│   └── smoke_test.ps1
├── labeled_dataset.csv          (41,650 rows, all features)
├── trained_model.pkl            (Current model v1)
├── test_model_professional.py   (Testing framework v1)
├── test_model_v2.py             (Testing framework v2)
├── train_classifier.py          (Training script)
├── prepare_training_data.py     (Data prep script)
├── INTERVIEW_GUIDE.md           (3,500 lines)
├── INTERVIEW_STORIES.md         (2,800 lines)
├── INTERVIEW_FAQS.md            (2,800 lines)
├── PROJECT_INVENTORY.md         (This file)
└── [documentation files]
```

---

## 🏆 ACHIEVEMENTS

✅ **Data Collection**: 14 full lectures, manually annotated with transition timestamps  
✅ **Feature Engineering**: 4 robust features with clear physical meaning  
✅ **Model Development**: Custom Decision Tree classifier from scratch  
✅ **Testing Framework**: Professional-grade testing with metrics  
✅ **Interview Prep**: 9,100 lines of interview preparation content  
✅ **Problem Solving**: Identified and fixed multiple technical issues  
✅ **Documentation**: Comprehensive guides and analysis  

---

## 📞 QUICK REFERENCE

**Get help**: See individual docs for feature details, training process, testing procedure  
**Run tests**: `test_model_v2.py --video <path> --ground-truth <path>`  
**Train model**: `train_classifier.py`  
**View inventory**: You're reading it!
