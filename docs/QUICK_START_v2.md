# Quick Start: Professional Testing Framework v2

## What is test_model_v2.py?

A professional-grade ML testing script with:
- ✅ Proper logging (not print statements)
- ✅ Type safety (full type hints)
- ✅ Structured data (dataclasses)
- ✅ Better error handling
- ✅ Cleaner output formatting

## Quick Commands

### 1. Test on training data (verify model works)
```bash
python test_model_v2.py --video "data/raw_videos/algorithms_14_hindi.mp4"
```
Expected: Detects ~40-50 transitions (this is a training video)

### 2. Test on new teacher with ground truth
```bash
python test_model_v2.py \
    --video "data/testing_videos/toc_1.mp4" \
    --ground-truth "data/testing_videos/toc_1_transitions.txt"
```
Output:
- Ground truth: 8 transitions
- Model detections: ? (to be seen)
- Metrics: Precision, Recall, F1-Score

### 3. Save results to file
```bash
python test_model_v2.py \
    --video "data/testing_videos/toc_1.mp4" \
    --ground-truth "data/testing_videos/toc_1_transitions.txt" \
    --output toc1_results.json
```
Creates JSON file with full results

### 4. Debug mode (verbose logging)
```bash
python test_model_v2.py \
    --video "data/testing_videos/toc_1.mp4" \
    --log-level DEBUG
```

## Output Explanation

### Without Ground Truth
```
[GROUND TRUTH]
No ground truth available

[PREDICTIONS]
Detections: 15
   1.   45.32s
   2.   67.89s
   ...
```

### With Ground Truth
```
[GROUND TRUTH]
Transitions: 8
   1.   1.22s
   2.   2.14s
   ...

[PREDICTIONS]
Detections: 5
   1.  1.20s
   2.  7.15s
   ...

[METRICS (5s tolerance)]
Confusion Matrix:
  TP (correct detections): 3
  FP (false alarms): 2
  FN (missed): 5

Performance:
  Precision: 60.0% (3 correct / 5 detected)
  Recall: 37.5% (3 correct / 8 actual)
  F1-Score: 0.4615

[ASSESSMENT]
Status: NEEDS IMPROVEMENT
  - Currently at 37% recall (target: 75%+)
  - Recommendations:
    1. Collect more training data
    2. Adjust feature engineering
    3. Fine-tune confidence threshold
```

## Understanding Metrics

### Confusion Matrix
- **TP (True Positives)**: Correctly detected transitions
- **FP (False Positives)**: Detected transitions that weren't actually there
- **FN (False Negatives)**: Missed transitions

### Performance Metrics
- **Precision = TP / (TP + FP)**
  - "Of the detected transitions, how many were correct?"
  - High precision = fewer false alarms
  
- **Recall = TP / (TP + FN)**
  - "Of the actual transitions, how many did we find?"
  - High recall = fewer missed transitions
  
- **F1-Score = 2 × (Precision × Recall) / (Precision + Recall)**
  - Balanced measure (0-1, higher is better)

## Tolerance Parameter

Default: 5 seconds

Example: If transition happens at 10.00s and model predicts 12.34s:
- Difference: 2.34 seconds
- Within 5s tolerance? YES → Counts as correct (TP)
- Within 2s tolerance? NO → Counts as missed (FN)

## Comparing v1 vs v2

### test_model_professional.py (v1)
```python
print("🎬 FRAME EXTRACTION")  # Uses emojis (causes encoding issues)
print(f"  ✅ Model loaded")    # Inconsistent formatting
# Ad-hoc metric calculation scattered through code
```

### test_model_v2.py (v2)
```python
logger.info("STEP 2: Extracting Frames")  # Structured logging
logger.info("Model loaded successfully")   # Consistent format
# Metrics calculated in dedicated class
metrics = EvaluationMetrics()
metrics.compute(predictions, ground_truth)
```

## File Outputs

### Option 1: Console only
```bash
python test_model_v2.py --video data/raw_videos/algorithms_14_hindi.mp4
```
Output: Printed to terminal

### Option 2: Save to JSON
```bash
python test_model_v2.py --video ... --output results.json
```
Output: `results.json` contains:
```json
{
  "timestamp": "2026-01-18T12:39:26.123456",
  "video": "data/raw_videos/algorithms_14_hindi.mp4",
  "fps": 1.0,
  "frames_processed": 1171,
  "predictions": [
    {
      "timestamp": 45.32,
      "confidence": 0.85,
      "frame_index": 1359,
      "features": {...}
    }
  ],
  "ground_truth": [1.22, 2.14, 4.24, ...],
  "metrics": {
    "true_positives": 5,
    "false_positives": 2,
    "false_negatives": 3,
    "precision": 0.714,
    "recall": 0.625,
    "f1_score": 0.667
  }
}
```

## Troubleshooting

### Command not found: python
```bash
.\.venv\Scripts\python.exe test_model_v2.py --video ...
```

### Model not found error
```
Error: Model not found: trained_model.pkl
```
Make sure you're in the project directory where `trained_model.pkl` exists.

### Out of memory on large videos
Reduce FPS sampling:
```bash
python test_model_v2.py --video data/raw_videos/algorithms_14_hindi.mp4 --fps 0.5
```
(Process every 2 seconds instead of 1)

## Next Videos to Test

1. **algorithms_14_hindi.mp4** - Training video (baseline)
2. **toc_1.mp4** - New teacher (8 transitions)
3. Your new video when uploaded
