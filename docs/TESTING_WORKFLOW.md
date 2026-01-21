# Model Testing & Training Workflow

## Current Status

- **Model State**: ❌ Needs retraining
- **Performance on toc_1**: 0% recall (detected 0/8 transitions)
- **Issue**: Model trained on older data, not generalizing to toc_1.mp4

## Professional Testing Workflow

### Step 1: Test Model on Video
Run this to evaluate model performance:

```bash
python test_model_professional.py
```

**Output:**
- Shows ground truth transitions (if available)
- Displays all detected transitions with confidence scores
- Compares predictions to ground truth (Precision, Recall, F1)
- Saves detailed JSON report

**When to use:**
- After training
- To compare different model versions
- To verify improvements

---

## Retraining Workflow

### Step 1: Prepare Training Data from toc_1
Convert your manual ground truth into training data:

```bash
python prepare_training_data.py
```

**What it does:**
1. Extracts frames from toc_1.mp4 at 1 FPS
2. Loads your 8 transitions from toc_1_transitions.txt
3. Computes features for all frames
4. Labels frames as transition/non-transition
5. Saves as `toc_1_training_data.csv`

**Output:**
- `toc_1_training_data.csv` - Labeled training data
- `data/frames/toc_1/` - Extracted frames

---

### Step 2: Merge Training Data
Combine toc_1 data with existing training data:

```bash
# Append to existing dataset
cat toc_1_training_data.csv >> labeled_dataset.csv
```

Or manually merge in Excel/Python:
```python
import pandas as pd

existing = pd.read_csv('labeled_dataset.csv')
new = pd.read_csv('toc_1_training_data.csv')
combined = pd.concat([existing, new], ignore_index=True)
combined.to_csv('labeled_dataset.csv', index=False)
```

---

### Step 3: Retrain Model
```bash
python train_classifier.py
```

**What happens:**
- Loads combined training data
- Trains Decision Tree with new data
- Evaluates on test set
- Saves `trained_model.pkl`

---

### Step 4: Test Improved Model
```bash
python test_model_professional.py
```

Check if recall improved! Target: **>75% recall**

---

## File Structure

```
├── train_classifier.py          # Train model
├── test_model_professional.py   # Test model (use this!)
├── prepare_training_data.py     # Prepare new training data
│
├── labeled_dataset.csv          # Training data (41K+ frames)
├── toc_1_training_data.csv      # New data from toc_1 (prep output)
├── trained_model.pkl            # Trained model (binary)
│
├── data/
│  └── testing_videos/
│     ├── toc_1.mp4
│     ├── toc_1_transitions.txt       # Your manual ground truth
│     ├── toc_1_model_predictions.txt # Model's predictions
│     └── toc_1_test_results.json    # Detailed test results
│
└── README.md (this file)
```

---

## Understanding Test Output

### Example Output:
```
GROUND TRUTH
─────────────────────────────────────────────────
  Transitions found: 8
    1. 1.25s (ideal frame: 1.15s)
    2. 3.13s (ideal frame: 2.54s)
    ... 6 more transitions

MODEL LOADING
─────────────────────────────────────────────────
  ✅ Model loaded successfully
     - Type: Decision Tree
     - Max depth: 15
     - Training normalization: True

FRAME EXTRACTION
─────────────────────────────────────────────────
  📹 Video Info:
     - FPS: 30.00
     - Total frames: 37,991
     - Sampling: every 30 frames
     - Extracted frames: 1,267

INFERENCE
─────────────────────────────────────────────────
  🤖 Running inference on 1,267 frames...
     - Positive predictions: 0 frames
     - Model predictions: ✅ Ready

POST-PROCESSING
─────────────────────────────────────────────────
  Detected transitions: 0

EVALUATION (5s tolerance)
─────────────────────────────────────────────────
  Confusion Matrix:
    TP (correct):     0
    FP (false alarm): 0
    FN (missed):      8

  Performance Metrics:
    Precision: 0.0%  
    Recall:    0.0%  
    F1-Score:  0.0000
```

### What the metrics mean:

- **Precision**: Of detected transitions, how many are correct?
  - Formula: TP / (TP + FP)
  - Example: If model finds 10 transitions and 8 are correct → 80% precision

- **Recall**: Of actual transitions, how many did model find?
  - Formula: TP / (TP + FN)
  - Example: If there are 8 real transitions and model finds 6 → 75% recall

- **F1-Score**: Balanced combination of precision and recall
  - Formula: 2 × (Precision × Recall) / (Precision + Recall)
  - Use this to compare different models

---

## Troubleshooting

### Model detects 0 transitions
**Possible causes:**
1. Model not trained on similar data
2. Video has different lighting/content than training data
3. Model features not capturing relevant information
4. Confidence threshold too high

**Solutions:**
1. Retrain with toc_1 data
2. Adjust feature engineering
3. Check training data quality
4. Experiment with different thresholds

### Model detects too many false positives (high FP)
**Solution:**
- Increase confidence threshold in `test_model_professional.py`
- Line: `detected = cluster_predictions(..., confidence_threshold=0.5)`

### Test takes too long
**Solution:**
- Reduce sampling rate (extract fewer frames)
- Change FPS in `test_model_professional.py`
- Line: `frames = extract_frames_from_video(video_path, fps=0.5)`

---

## Why Model Needs Retraining

Your original training data was from 14 videos with specific characteristics. The toc_1 video:
- May have different lighting conditions
- May have different slide content layout
- May have different transition speed
- Features may be distributed differently

**Solution**: Add toc_1 to training data → retrain → test again

---

## Quick Commands Reference

```bash
# Full workflow
python prepare_training_data.py      # Prepare toc_1 data
python train_classifier.py           # Retrain with combined data
python test_model_professional.py    # Test on toc_1
```

---

## Next Steps

1. ✅ You've verified there are **8 transitions** in toc_1
2. ⏳ **Run**: `python prepare_training_data.py`
3. ⏳ **Merge** training data
4. ⏳ **Run**: `python train_classifier.py`
5. ⏳ **Run**: `python test_model_professional.py`
6. ✅ **Check** if recall improved
7. 🔄 **Repeat** if needed
