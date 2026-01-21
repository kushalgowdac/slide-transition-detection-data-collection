# ⚡ QUICK START: Build Model v2 in 20 Minutes

## 🎯 Your Current Situation

✅ **What works**: Model v1 (97.45% accuracy on training videos)  
❌ **What doesn't**: Model v1 fails on new teachers (0% recall on algo_1, cn_1)  
🔧 **Problem**: Training data is biased (84.4% train, 6.7% test - should be 70/30)  
✅ **Solution**: Build Model v2 with proper stratified data

---

## 🚀 EXECUTION (20 minutes total)

### Step 1: Create Balanced Dataset (5 min)

```bash
.\.venv\Scripts\python.exe create_stratified_dataset_v2.py
```

**What it does**:
- Reads `labeled_dataset.csv` (biased)
- Splits each video 70/30 (no data leakage)
- Balances transition ratio (~2.4% in both splits)
- Writes `labeled_dataset_v2.csv` (balanced)

**Expected output**:
```
CREATING STRATIFIED SPLIT (v2)
Processing 14 videos...
  [1] algorithms_14_hindi: Total 1,187 → Train 830 + Test 357
  [2] chemistry_01_english: Total 10,626 → Train 7,438 + Test 3,188
  ...
Total TRAIN: 29,400 rows (70%)
Total TEST: 11,400 rows (30%)

Train transition %: 2.4%
Test transition %: 2.4%
✅ Balance check: EXCELLENT balance (diff: 0.02%)
```

### Step 2: Train Model v2 (3 min)

```bash
.\.venv\Scripts\python.exe train_classifier_v2.py
```

**What it does**:
- Loads `labeled_dataset_v2.csv` (balanced)
- Trains Decision Tree on 70% data
- Evaluates on 30% data
- Saves `trained_model_v2.pkl`

**Expected output**:
```
TRAINING MODEL v2
Training Decision Tree with max_depth=15...
✅ Model trained successfully

EVALUATING MODEL v2
Test Set Performance:
  - Accuracy:  0.9745 (97.45%)
  - Precision: 0.7725 (77.25%)
  - Recall:    0.7963 (79.63%)
  - F1-Score:  0.7842

✅ MODEL v2 TRAINING COMPLETE!
```

### Step 3: Test algo_1 with Model v2 (5 min)

```bash
.\.venv\Scripts\python.exe test_model_v2.py \
  --video "data/testing_videos/algo_1.mp4" \
  --ground-truth "data/testing_videos/algo_1_transitions.txt" \
  --model "trained_model_v2.pkl" \
  --fps 1.0
```

**What it does**:
- Loads new model
- Extracts frames from algo_1
- Runs inference
- Compares to ground truth
- Shows metrics

**What to watch for**:
```
GROUND TRUTH: 10 transitions found
PREDICTIONS: ??? (v1 found 387, should be better with v2)
METRICS: Precision, Recall, F1-Score
```

### Step 4: Test cn_1 with Model v2 (5 min)

```bash
.\.venv\Scripts\python.exe test_model_v2.py \
  --video "data/testing_videos/cn_1.mp4" \
  --ground-truth "data/testing_videos/cn_1_transitions.txt" \
  --model "trained_model_v2.pkl" \
  --fps 1.0
```

### Step 5: Compare v1 vs v2 Results (2 min)

**Create comparison**:

| Model | algo_1 Recall | cn_1 Recall | Generalization |
|-------|---------------|------------|-----------------|
| v1 | 0% ❌ | 0% ❌ | Poor |
| v2 | ??? | ??? | Testing... |

**If v2 improves**:
- → Use v2 as new production model
- → Train v2 on all data (with best hyperparameters)

**If v2 doesn't improve**:
- → Analyze feature engineering
- → Try different hyperparameters
- → Collect more training data

---

## 📊 KEY DIFFERENCES: v1 vs v2

### Data Split
```
v1 (BIASED):           v2 (BALANCED):
├─ Train: 84.4%        ├─ Train: 70%
├─ Val: 8.9%           └─ Test: 30%
└─ Test: 6.7%

v1 Test: 5.8% transitions ← 2.7x more than training!
v2 Test: 2.4% transitions ← Same as training ✅
```

### Video Distribution
```
v1 (BIASED):           v2 (BALANCED):
├─ chemistry_04: 31.9% ← DOMINATES
├─ chemistry_01: 25.5% ← DOMINATES
└─ Others: 42.6%       ├─ Balanced across all 14
                       └─ Each teacher ~7%
```

### Files
```
v1 Files:              v2 Files:
├─ labeled_dataset.csv ├─ labeled_dataset_v2.csv (NEW)
├─ trained_model.pkl   ├─ trained_model_v2.pkl (NEW)
└─ [test results]      └─ [test results]
```

---

## 🔍 WHAT TO EXPECT

### Before Running
- Model v1: Fails on new videos (0% recall)
- Why: Trained heavily on chemistry lectures

### After Running v2
- **Best case**: v2 detects some transitions in algo_1/cn_1 ✅
  - Shows proper stratification helped
  - Indicates better generalization
  
- **Worst case**: v2 still fails (0% recall)
  - Means data bias wasn't the issue
  - Might need more training data from diverse teachers
  - Or feature engineering improvements

---

## ⚙️ AVAILABLE OPTIONS

### If You Want to Adjust v2 During Creation

Edit `create_stratified_dataset_v2.py`:
```python
TRAIN_SIZE = 0.70  # Change to 0.80 for 80/20 split
TEST_SIZE = 0.30   # Adjust accordingly
```

Edit `train_classifier_v2.py`:
```python
MAX_DEPTH = 15  # Reduce to 10 for smaller tree, increase to 20 for larger
```

---

## 📚 FILE REFERENCES

**Just Created**:
- `PROJECT_INVENTORY.md` - Complete overview of all work
- `MODEL_v2_STRATEGY.md` - Detailed v2 roadmap
- `COMPLETE_PROJECT_SUMMARY.md` - This summary
- `create_stratified_dataset_v2.py` - Dataset creator script
- `train_classifier_v2.py` - Model trainer script

**Already Exist**:
- `test_model_v2.py` - Testing framework (works with both v1 and v2)
- `INTERVIEW_GUIDE.md` - Interview prep (3,500 lines)
- `INTERVIEW_STORIES.md` - Narratives (2,800 lines)
- `INTERVIEW_FAQS.md` - Q&A (2,800 lines)

---

## 💡 WHY THIS APPROACH?

1. **Non-destructive**: Original model v1 stays safe
2. **Honest comparison**: Side-by-side testing possible
3. **Best practice**: Proper train/test split per ML standards
4. **Learning**: Understand why stratification matters
5. **Incremental**: Clear before/after metrics

---

## 🎯 SUCCESS METRICS

**v2 is successful if**:
- ✅ Dataset creation completes (29,400 train + 11,400 test)
- ✅ Model trains without errors
- ✅ Class balance is ~2.4% in both splits (not 2.2% vs 5.8%)
- ✅ Recall on algo_1 > 0% (currently fails completely)
- ✅ Recall on cn_1 > 0% (currently fails completely)

---

## 🚨 TROUBLESHOOTING

**If stratified dataset creation fails**:
- Check `labeled_dataset.csv` exists and is readable
- Ensure you're in project root directory
- Check Python 3.13.7 is active: `.\.venv\Scripts\python.exe --version`

**If model training fails**:
- Check `labeled_dataset_v2.csv` was created
- Ensure `src/classifier.py` exists (or copy from v1 setup)
- Check disk space (model ~15KB, dataset ~5MB)

**If testing fails**:
- Check video file exists: `data/testing_videos/algo_1.mp4`
- Check ground truth file: `data/testing_videos/algo_1_transitions.txt`
- Check model file: `trained_model_v2.pkl`

---

## 📞 WHEN YOU'RE DONE

1. Compare v1 vs v2 metrics
2. Update `MODEL_v2_RESULTS.md` with findings
3. Decide: Keep v1, use v2, or combine approaches?
4. Archive this work with results

---

**Ready to start? Execute:**
```bash
.\.venv\Scripts\python.exe create_stratified_dataset_v2.py
```

**Total Time: ~20 minutes**

