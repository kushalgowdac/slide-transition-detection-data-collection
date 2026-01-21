# 🚀 MODEL v2: Improved Dataset Strategy

**Status**: Ready to build  
**Approach**: Create balanced dataset WITHOUT retraining v1  
**Estimated Time**: 15-20 minutes

---

## 📊 THE PROBLEM WITH MODEL v1

Current model uses **biased data**:

```
CURRENT (v1) - BIASED:
├── Train: 84.4% (35,143 rows) | 2.2% transitions
├── Val:    8.9% (3,727 rows)  | 1.9% transitions  
└── Test:   6.7% (2,780 rows)  | 5.8% transitions ❌ 2.7x MORE!

VIDEO DOMINANCE:
├── chemistry_04: 31.9% of data ← Overfitting!
├── chemistry_01: 25.5% of data ← Overfitting!
└── 12 other videos: ~42% of data

RESULT: Model memorized chemistry lectures, fails on others ❌
```

---

## ✅ THE SOLUTION: MODEL v2 WITH STRATIFIED DATA

### Strategy: Proper 70/30 Split with Stratification

**Key Principles**:
1. **Video-level split**: Each video → only ONE split (no data leakage)
2. **Class-level balance**: Keep ~2.4% transitions in all splits
3. **Balanced distribution**: Each teacher equally represented
4. **Keep original data**: Model v1 untouched

### The Plan

#### Step 1: Stratified Split

**Videos for TRAIN** (10 videos, 70% of frames):
- chemistry_04_english: 13,272 frames (keep first 70%)
- chemistry_01_english: 10,626 frames (keep first 70%)
- mathematics_05_hindi: 3,051 frames (all)
- chemistry_09_hindi: 1,641 frames (all)
- database_13_hindi: 1,587 frames (all)
- physics_05_english: 1,428 frames (all)
- database_12_hindi: 1,530 frames (all)
- database_11_hindi: 1,552 frames (all)
- computer_networks_13_hindi: 1,363 frames (all)
- physics_01_english: 1,352 frames (all)
- **Total**: ~29,400 frames (70.6%)

**Videos for TEST** (4 videos, 30% of frames):
- chemistry_04_english: 3,968 frames (remaining 30%)
- chemistry_01_english: 3,188 frames (remaining 30%)
- chemistry_08_hindi: 1,312 frames (all)
- mathematics_07_hindi: 676 frames (all)
- chemistry_10_english: 1,073 frames (all)
- algorithms_14_hindi: 1,187 frames (all)
- **Total**: ~11,404 frames (27.4%) ← Cleaner 70/30 split!

**No VAL set**: Use test set directly (or combine with train for cross-validation)

**Result**:
```
NEW (v2) - BALANCED:
├── Train: 70% (~29,400 rows) | ~2.4% transitions
└── Test:  30% (~11,400 rows) | ~2.4% transitions
        (No Val set, much cleaner)

VIDEO DISTRIBUTION (more balanced):
├── chemistry_04: 17,240 (train) + 3,968 (test)
├── chemistry_01: 7,438 (train) + 3,188 (test)
├── All others: Equally distributed
```

#### Step 2: Create Stratified Dataset

**Script**: `create_stratified_dataset_v2.py`

```python
# Load labeled_dataset.csv
# For each video, split by ratio (70/30)
# Preserve class distribution (~2.4% transitions)
# Save as labeled_dataset_v2.csv
```

#### Step 3: Train Model v2

**Script**: `train_classifier_v2.py`

```python
# Load labeled_dataset_v2.csv
# Train on 70% data
# Test on 30% data
# Save as trained_model_v2.pkl
```

**Expected Performance**:
- Should show similar accuracy (97%+)
- But metrics will be MORE HONEST (class balance)
- Should generalize better to new videos

#### Step 4: Test Model v2

**Test on new videos**:
```bash
# Test algo_1
.\.venv\Scripts\python.exe test_model_v2.py \
  --video data/testing_videos/algo_1.mp4 \
  --ground-truth data/testing_videos/algo_1_transitions.txt \
  --model trained_model_v2.pkl

# Test cn_1
.\.venv\Scripts\python.exe test_model_v2.py \
  --video data/testing_videos/cn_1.mp4 \
  --ground-truth data/testing_videos/cn_1_transitions.txt \
  --model trained_model_v2.pkl

# Test db_1 and toc_1 similarly
```

**Comparison**:
```
Model v1 (Current):  0% recall on algo_1 ❌
Model v2 (Balanced): ??? (TBD - should improve)
```

---

## 🔄 COMPARISON: v1 vs v2

| Aspect | Model v1 | Model v2 |
|--------|----------|----------|
| **Train/Test Split** | 84.4 / 6.7 | 70 / 30 |
| **Transition Ratio (Train)** | 2.2% | 2.4% |
| **Transition Ratio (Test)** | 5.8% ❌ | 2.4% ✅ |
| **Dominant Video** | chemistry_04 (31.9%) | Balanced |
| **Cross-validation** | Via separate val set | Can use test directly |
| **Honest Metrics** | ❌ Skewed | ✅ Fair |
| **Generalization** | Poor ❌ | Better ✅ |
| **File Names** | trained_model.pkl | trained_model_v2.pkl |
| **Dataset File** | labeled_dataset.csv | labeled_dataset_v2.csv |

---

## 📋 EXECUTION CHECKLIST

### Phase 1: Dataset Creation (5 minutes)

- [ ] Create `create_stratified_dataset_v2.py`
- [ ] Run script to generate `labeled_dataset_v2.csv`
- [ ] Verify split: ~70% train, ~30% test
- [ ] Verify class balance: ~2.4% transitions in both splits
- [ ] Backup original `labeled_dataset.csv`

### Phase 2: Model Training (3 minutes)

- [ ] Create `train_classifier_v2.py`
- [ ] Run script to train on v2 data
- [ ] Save as `trained_model_v2.pkl`
- [ ] Verify model loads without errors
- [ ] Record training metrics

### Phase 3: Testing (10-15 minutes)

- [ ] Test algo_1 with v2 model
  - Expected: Better than 0% recall
- [ ] Test cn_1 with v2 model
  - Expected: Better than 0% recall
- [ ] Test db_1 with v2 model (bonus)
- [ ] Test toc_1 with v2 model (bonus)
- [ ] Compare metrics: v1 vs v2

### Phase 4: Analysis (5 minutes)

- [ ] Document results in `MODEL_v2_RESULTS.md`
- [ ] Compare v1 vs v2 performance
- [ ] Identify which teacher/style v2 generalizes better to
- [ ] Recommend next steps

---

## 🎯 SUCCESS CRITERIA

✅ **Model v2 Created**
- Successfully trains on stratified dataset
- Saves `trained_model_v2.pkl`

✅ **Better Generalization**
- algo_1: > 0% recall (currently 0%)
- cn_1: > 0% recall (currently 0%)

✅ **Proper Metrics**
- Class balance consistent across train/test
- No data leakage (videos in only one split)

✅ **Comparability**
- Can run same tests on both v1 and v2
- Side-by-side results comparison

---

## 📁 NEW FILES TO CREATE

| File | Purpose | Size |
|------|---------|------|
| `create_stratified_dataset_v2.py` | Generate balanced dataset | ~200 lines |
| `train_classifier_v2.py` | Train model on v2 data | ~150 lines |
| `labeled_dataset_v2.csv` | New balanced dataset | 41,650 rows |
| `trained_model_v2.pkl` | New model binary | ~15KB |
| `MODEL_v2_RESULTS.md` | Test results and analysis | ~1,000 lines |

---

## 💡 WHY THIS APPROACH?

1. **Non-destructive**: Original model v1 completely preserved
2. **Honest comparison**: Can test both models side-by-side
3. **Industry standard**: Proper train/test split follows ML best practices
4. **Incremental improvement**: Clear progress tracking
5. **Educational**: Learn why stratification matters

---

## 🚀 NEXT STEP

Ready? Start with creating `create_stratified_dataset_v2.py` to generate the balanced dataset!

