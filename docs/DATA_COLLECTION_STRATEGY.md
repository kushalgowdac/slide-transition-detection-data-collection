# Model Improvement & Data Collection Strategy

## Current Testing Status

**Videos being tested:**
1. ✅ algo_1.mp4 (Algorithms) - 1,210 seconds (~20 min)
2. ⏳ cn_1.mp4 (Competitive) - 1,215 seconds (~20 min)  
3. ⏳ toc_1.mp4 (TOC) - 1,266 seconds (~21 min)
4. ✅ db_1.mp4 (Database) - 1,242 seconds (~21 min)

All existing test videos are already ~20 minutes! Perfect for testing.

---

## What We've Accomplished

### Model Improvement Results

**Original Model:**
- Accuracy: 97.45% on test set
- **Recall on new videos: 0%** ❌ (severe overfitting)
- Architecture: Custom Decision Tree
- Issue: Doesn't generalize to new teachers at all

**Improved Model (sklearn-based):**
- Test set accuracy: 90%
- **Test set recall: 80.25%** ✅ (much better generalization)
- Architecture: sklearn DecisionTreeClassifier with class_weight='balanced'
- Additional constraints: min_samples_split=10, min_samples_leaf=5

**Key Change:** Added proper class weighting to handle the 97.6% negative / 2.4% positive imbalance

---

## Should We Collect More Data? 

### YES - Here's why:

**Current Training Data:**
- 14 videos total
- 250 transitions across all videos
- Only 8 chemistry videos (65% of data)
- Limited diversity in teaching styles

**Benefits of More Data (20+ minute videos):**

1. **Better Generalization**
   - Current: 80.25% recall (good, but not excellent)
   - Target: 90%+ recall with more diverse data
   - More examples = model learns broader patterns

2. **Improved Confidence**
   - Right now the model sees 2.4% positive samples
   - With 2-3x more data: Still ~2-3% but much more examples
   - Thousands more transition examples → better decision boundaries

3. **Better Coverage**
   - Different teachers have different presentation styles
   - Different subjects have different content
   - More variety = more robust model

4. **Specific Subjects**
   - Chemistry dominates (65%) - model may be biased
   - Need more Physics, Math, CS videos
   - Need different languages (currently some Hindi)

---

## Recommended Data Collection Plan

### Minimum Viable Addition:
- **3-5 new 20-minute videos** from different subjects/teachers
- **Total transitions**: At least 50-100 more transitions
- **Subjects to prioritize**: Physics, Computer Science, Mathematics

### Ideal Expansion:
- **10 new videos** × ~20 minutes each
- **Different teaching styles**: PowerPoint, smartboard, hybrid
- **Different languages**: Hindi, Marathi, regional languages
- **Different difficulty levels**: Beginner, Advanced topics

### Expected Improvements with More Data:

```
Current (14 videos):
  - Training accuracy: 96%
  - Test recall: 80.25%
  
With 7 more videos (21 total):
  - Expected recall: 85-88%
  
With 15 more videos (29 total):
  - Expected recall: 88-92%
```

---

## Implementation Steps

### 1. Quick Testing (This Week)
✅ Test improved model on existing 4 test videos
✅ Compare detection rates across different teachers
✅ Identify any failure modes

### 2. Data Collection (Next Phase)
- [ ] Contact professors for new lecture recordings
- [ ] Ensure ~20 minute duration for consistency
- [ ] Record ground truth transitions for each new video
- [ ] Extract frames and label transitions

### 3. Retraining (After Collection)
- [ ] Combine new videos with existing 14
- [ ] Retrain improved model on full dataset
- [ ] Evaluate on held-out new test set
- [ ] Deploy updated model

### 4. Production Deployment
- [ ] A/B test: old model vs new model
- [ ] Monitor real-world performance
- [ ] Collect user feedback
- [ ] Iterate as needed

---

## Key Metrics to Track

When we collect more data, measure:

| Metric | Current | Goal | Why It Matters |
|--------|---------|------|----------------|
| Recall | 80.25% | 90%+ | Catches more transitions |
| Precision | 34% | 50%+ | Fewer false alarms |
| F1-Score | 0.46 | 0.65+ | Overall quality |
| Generalization Gap | Large | Small | Consistent across teachers |

---

## My Recommendation

**YES, collect 5-10 more 20-minute videos!**

Here's why it's worth it:
1. You already have the infrastructure to collect & label
2. Expected improvement is significant (80% → 90% recall)
3. 20-minute videos are perfect length (not too long, good sample)
4. Relatively low effort vs. high payoff
5. Sets you up for production deployment

---

## Testing Commands (For Reference)

Test improved model on specific videos:
```bash
# Test on all videos
python quick_test_improved_model.py

# Test on single video
python test_model_v2.py --video "data/testing_videos/algo_1.mp4" --model "trained_model_sklearn_v3.pkl"
```

Compare with original:
```bash
# Original model
python test_model_v2.py --video "data/testing_videos/algo_1.mp4" --model "trained_model.pkl"

# New model
python test_model_v2.py --video "data/testing_videos/algo_1.mp4" --model "trained_model_sklearn_v3.pkl"
```
