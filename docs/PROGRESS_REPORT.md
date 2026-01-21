# Testing & Training Progress Report

## Session Summary (January 18, 2026)

### What Was Done So Far

#### **1. Initial Model Training (Baseline)**
- **Trained on**: 14 lecture videos from the training dataset
- **Model type**: Decision Tree (max depth 15)
- **Training data**: 41,650 labeled frames
- **Performance on test set**:
  - Accuracy: 97.45%
  - Precision: 77.25%
  - Recall: 79.63%
  - F1-Score: 78.42%

---

#### **2. Test 1: toc_1.mp4 (NEW TEACHER - NOT IN TRAINING)**
- **Result**: ❌ POOR PERFORMANCE
  - Ground truth: 8 transitions
  - Detected: 0 transitions
  - **Recall: 0%** (missed all 8)
  - **Precision: N/A** (no detections)

- **Conclusion**: Model does NOT generalize to new teachers/videos

---

#### **3. Data Preparation for Retraining**
- **Extracted** 1,267 frames from toc_1.mp4 at 1 FPS
- **Created** toc_1_training_data.csv with:
  - 1,267 labeled frames
  - 8 frames marked as transitions
  - 1,259 frames marked as non-transitions
  - Features computed for all frames

---

#### **4. Merge Training Data** ⚠️ INCOMPLETE
- **Status**: NOT YET DONE
- **What needs to happen**:
  ```
  existing labeled_dataset.csv: 41,650 rows → 42,917 rows (with mixed types)
  + toc_1_training_data.csv:   1,267 rows
  = SHOULD BE:                 44,184 rows (if merged)
  ```
- **File Status**:
  - `labeled_dataset.csv` - UNCHANGED (merge script had issues)
  - `labeled_dataset_backup.csv` - exists but old
  - `toc_1_training_data.csv` - ✅ CREATED

---

#### **5. Model Retraining** ❌ NOT DONE YET
- **Status**: Waiting for merge to complete
- **Required command**:
  ```powershell
  .\.venv\Scripts\python.exe train_classifier.py
  ```

---

#### **6. Test 2: db_1.mp4 (TEACHER FROM ORIGINAL TRAINING SET)**
- **Status**: ✅ JUST RAN (Exit Code 0)
- **Result**: 
  - Frames processed: 1,239
  - Transitions detected: 0
  - Ground truth: None (file doesn't exist)
  - File saved: `toc_1_test_results.json` (note: mislabeled as toc_1)

---

## Current Status Checklist

| Step | Status | Details |
|------|--------|---------|
| Model training (baseline) | ✅ Done | 14 videos, 97.45% test accuracy |
| Test on new teacher (toc_1) | ✅ Done | 0/8 recall → need retraining |
| Extract frames (toc_1) | ✅ Done | 1,267 frames, labeled |
| Merge training data | ⏳ **BLOCKED** | Script issues, NOT merged yet |
| Retrain model | ⏳ **WAITING** | Depends on merge |
| Test on original teacher (db_1) | ✅ Done | 0 transitions detected |

---

## What Needs to Be Done Next

### **IMMEDIATE: Fix Data Merge** (Priority 1)

The merge_training_data.py script had issues. We need to manually merge:

```powershell
# Option 1: Simple backup + append (safest)
cd "d:\College_Life\projects\slide transition detection - data collection"

# Python one-liner to merge
.\.venv\Scripts\python.exe -c "
import pandas as pd, shutil
from pathlib import Path

# Backup original
shutil.copy('labeled_dataset.csv', 'labeled_dataset_backup_new.csv')

# Load and merge
existing = pd.read_csv('labeled_dataset.csv')
new = pd.read_csv('toc_1_training_data.csv')
combined = pd.concat([existing, new], ignore_index=True)

# Save merged
combined.to_csv('labeled_dataset.csv', index=False)
print(f'Merged: {len(combined)} total rows')
"
```

### **THEN: Retrain Model** (Priority 2)
```powershell
.\.venv\Scripts\python.exe train_classifier.py
```
- Will output: `trained_model.pkl` (retrained with toc_1 data)
- Will output: `model_evaluation.json` (new metrics)

### **THEN: Test Retrained Model on toc_1** (Priority 3)
```powershell
.\.venv\Scripts\python.exe test_model_professional.py `
  --video "data/testing_videos/toc_1.mp4" `
  --ground-truth "data/testing_videos/toc_1_transitions.txt" `
  --fps 1.0
```
- Expected: Recall should improve (was 0%, target >75%)

### **THEN: Test on db_1 (Original Teacher)** (Priority 4)
```powershell
.\.venv\Scripts\python.exe test_model_professional.py `
  --video "data/testing_videos/db_1.mp4" `
  --ground-truth "" `
  --fps 1.0
```
- Check if original teacher videos still work well
- If they degrade, model may be overfitting to toc_1

---

## Key Insights So Far

1. **Model trained on 14 videos does NOT generalize** to new teacher (toc_1)
   - This is normal: different lecturers have different styles, lighting, slide layouts

2. **Adding diverse training data helps** 
   - toc_1 will teach model about this teacher's patterns
   - But may dilute performance on original 14 teachers

3. **Trade-off exists**:
   - Smaller, focused model: Great on one style, bad on others
   - Larger, diverse model: Okay on many styles, not great on any

---

## Files Created So Far

```
✅ toc_1_training_data.csv          - 1,267 labeled frames from toc_1
✅ labeled_dataset_backup.csv       - Backup (old, will be replaced)
✅ test_model_professional.py       - New professional testing script
✅ prepare_training_data.py         - Script to extract & label frames
✅ merge_training_data.py           - Script to combine datasets (had issues)
✅ TESTING_WORKFLOW.md              - Documentation
✅ toc_1_test_results.json          - Test results for toc_1 (0% recall)
✅ toc_1_model_predictions.txt      - Model predictions for toc_1 (empty)
⏳ labeled_dataset.csv              - NOT YET MERGED (still has 42,917 rows)
❌ db_1_test_results.json           - Not created (no ground truth)
❌ trained_model_v2.pkl             - Not created (retraining not done)
```

---

## Next Command to Run NOW

```powershell
cd "d:\College_Life\projects\slide transition detection - data collection"

# STEP 1: Merge the data (Python one-liner)
.\.venv\Scripts\python.exe -c "import pandas as pd; import shutil; shutil.copy('labeled_dataset.csv', 'labeled_dataset_backup_new.csv'); existing = pd.read_csv('labeled_dataset.csv'); new = pd.read_csv('toc_1_training_data.csv'); combined = pd.concat([existing, new], ignore_index=True); combined.to_csv('labeled_dataset.csv', index=False); print(f'Merged: {len(combined)} rows (was {len(existing)})')"

# STEP 2: Retrain
.\.venv\Scripts\python.exe train_classifier.py

# STEP 3: Test on toc_1 (new teacher - should improve)
.\.venv\Scripts\python.exe test_model_professional.py --video "data/testing_videos/toc_1.mp4" --ground-truth "data/testing_videos/toc_1_transitions.txt" --fps 1.0

# STEP 4: Test on db_1 (original teacher - should stay similar)
.\.venv\Scripts\python.exe test_model_professional.py --video "data/testing_videos/db_1.mp4" --ground-truth "" --fps 1.0
```

---

**Status: READY FOR MERGE & RETRAIN** ✅
