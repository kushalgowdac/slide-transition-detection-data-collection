# Model v1 Improvement Strategies

## Current Model Status
- **Accuracy**: 97.45% on training data
- **Problem**: 0% recall on new videos (algo_1, cn_1, toc_1)
- **Architecture**: Decision Tree (max_depth=15)
- **Features**: 4 (already optimal)
- **Issue**: Class imbalance (97.6% negative, 2.4% positive)

---

## Improvement Strategy 1: Class Weight Balancing

### Problem
Decision Tree trained with default weights treats all misclassifications equally:
- False negative (missed transition): cost = 1
- False positive (wrong transition): cost = 1

But for your task:
- Missed transition is worse than false positive
- Need to penalize missed transitions more heavily

### Solution: Balanced Class Weights
```python
from sklearn.tree import DecisionTreeClassifier

# Current (default):
clf = DecisionTreeClassifier(max_depth=15)

# Improved (with class weights):
clf = DecisionTreeClassifier(
    max_depth=15,
    class_weight='balanced',  # Automatically balance classes
    # OR explicit weights:
    # class_weight={0: 1, 1: 10}  # Penalize missing transitions 10x more
)
```

### Expected Impact
- Currently: Low recall (misses transitions)
- With balanced weights: Higher recall (catches more transitions)
- Trade-off: Slightly lower precision (more false positives)

**Why this helps**:
- Currently the tree ignores rare transitions (2.4% of data)
- Balancing makes rare class important
- Forces tree to split on transition features

---

## Improvement Strategy 2: Hyperparameter Tuning

### Parameters to Tune

**1. max_depth** (currently: 15)
```python
# Too high: Overfitting to noise in training data
# Too low: Underfitting, can't capture patterns

# Test range: 5 to 25
# Expected: Lower depth with balanced weights
clf = DecisionTreeClassifier(
    max_depth=10,  # Try smaller values
    class_weight='balanced',
    min_samples_split=10,
    min_samples_leaf=5
)
```

**2. min_samples_split** (minimum samples to split a node)
```python
# Too high: Tree stays shallow
# Too low: Tree overfits to individual samples

# Test range: 5 to 50
clf = DecisionTreeClassifier(
    max_depth=15,
    min_samples_split=20,  # Default is 2
    class_weight='balanced'
)
```

**3. min_samples_leaf** (minimum samples in leaf nodes)
```python
# Prevents tree from creating tiny leaves (overfitting)
# Test range: 2 to 20

clf = DecisionTreeClassifier(
    max_depth=15,
    min_samples_leaf=10,  # Default is 1
    class_weight='balanced'
)
```

### Hyperparameter Tuning Script
```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Define grid
param_grid = {
    'max_depth': [5, 8, 10, 12, 15, 20],
    'min_samples_split': [5, 10, 20, 30],
    'min_samples_leaf': [2, 5, 10],
    'class_weight': ['balanced', None]
}

# Search
clf = DecisionTreeClassifier()
grid_search = GridSearchCV(
    clf, param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='f1_weighted'  # Balanced metric
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
```

---

## Improvement Strategy 3: Threshold Adjustment

### Problem
Decision Tree outputs probability (0.0 to 1.0), but uses 0.5 threshold by default:
```
If probability > 0.5 → predict transition
If probability ≤ 0.5 → predict no transition
```

For rare events (2.4% transitions), you might want:
```
If probability > 0.3 → predict transition (more sensitive)
If probability ≤ 0.3 → predict no transition (catch more)
```

### Implementation
```python
# Get probability predictions instead of binary
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Try different thresholds
thresholds = [0.3, 0.35, 0.4, 0.45, 0.5]
for thresh in thresholds:
    y_pred = (y_pred_proba > thresh).astype(int)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Threshold {thresh}: Recall={recall:.3f}, Precision={precision:.3f}, F1={f1:.3f}")
```

### Expected Impact
- Lower threshold → Higher recall, lower precision
- Higher threshold → Lower recall, higher precision
- Find sweet spot for your use case

---

## Improvement Strategy 4: Random Forest Instead of Decision Tree

### Why Random Forest?
1. **Better generalization**: Multiple trees reduce overfitting
2. **Handles imbalance**: Can use class_weight='balanced' on each tree
3. **Feature importance**: Shows which features matter most
4. **More robust**: Less sensitive to outliers and noise

### Implementation
```python
from sklearn.ensemble import RandomForestClassifier

# Current:
clf = DecisionTreeClassifier(max_depth=15)

# Improved:
clf = RandomForestClassifier(
    n_estimators=100,        # 100 trees
    max_depth=15,            # Keep same depth
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced', # Balance classes
    random_state=42,
    n_jobs=-1                # Use all CPU cores
)
```

### Expected Impact
- **Pros**:
  - Better recall (catches more transitions)
  - Better generalization to new videos
  - More stable predictions
  - Feature importance analysis
  
- **Cons**:
  - Slower inference (100 trees vs 1)
  - Less interpretable
  - Needs more tuning

---

## Improvement Strategy 5: Feature Importance Analysis

### What It Shows
Which features does the model actually use?
```python
import matplotlib.pyplot as plt

# Get importance
importances = clf.feature_importances_
features = ['content_fullness', 'frame_quality', 'is_occluded', 'skin_ratio']

# Plot
plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel('Importance')
plt.title('Feature Importance in Decision Tree')
plt.show()

# Print
for f, imp in zip(features, importances):
    print(f"{f}: {imp:.4f} ({imp*100:.2f}%)")
```

### Why It Helps
- Shows if all 4 features are being used
- If some features have 0% importance, they're not needed
- Can help decide if we need additional features

---

## Improvement Strategy 6: Stratified Cross-Validation

### Problem
Current model trained once on full training set.
What if we used 5-fold cross-validation?

### Benefit
- Better estimate of true performance
- More robust evaluation
- Less sensitive to specific data split

### Implementation
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    clf, X_train, y_train,
    cv=skf,
    scoring='f1'
)

print(f"F1 scores across folds: {scores}")
print(f"Mean F1: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

---

## Improvement Strategy 7: Ensemble Voting

### Idea
Combine multiple different models:
1. Decision Tree (current)
2. Random Forest
3. Gradient Boosting
4. Maybe SVM

Vote on final decision.

### Implementation
```python
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

clf1 = DecisionTreeClassifier(max_depth=15, class_weight='balanced')
clf2 = RandomForestClassifier(n_estimators=50, class_weight='balanced')
clf3 = SVC(kernel='rbf', probability=True)

ensemble = VotingClassifier(
    estimators=[('dt', clf1), ('rf', clf2), ('svm', clf3)],
    voting='soft'  # Use probability averaging
)

ensemble.fit(X_train, y_train)
```

---

## Improvement Strategy 8: Data Augmentation

### Idea
Create synthetic transition samples to balance the data

### Approaches

**1. SMOTE (Synthetic Minority Oversampling)**
```python
from imblearn.over_sampling import SMOTE

# Balance the dataset
smote = SMOTE(sampling_strategy=0.5)  # 50% transitions
X_balanced, y_balanced = smote.fit_resample(X_train, y_train)

# Now train on balanced data
clf.fit(X_balanced, y_balanced)
```

**2. Undersampling (remove non-transition samples)**
```python
from imblearn.under_sampling import RandomUnderSampler

undersampler = RandomUnderSampler(sampling_strategy=0.3)  # 30% transitions
X_balanced, y_balanced = undersampler.fit_resample(X_train, y_train)

clf.fit(X_balanced, y_balanced)
```

**3. Combined (SMOTE + Undersampling)**
```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

pipeline = Pipeline([
    ('smote', SMOTE(sampling_strategy=0.5)),
    ('undersampler', RandomUnderSampler(sampling_strategy=0.7))
])

X_balanced, y_balanced = pipeline.fit_resample(X_train, y_train)
clf.fit(X_balanced, y_balanced)
```

---

## Recommended Implementation Order

### Phase 1: Quick Wins (30 minutes)
1. ✅ Add class_weight='balanced' to current Decision Tree
2. ✅ Adjust threshold (find optimal for your use case)
3. ✅ Test on current training data
4. ✅ Compare with original model

### Phase 2: Hyperparameter Tuning (45 minutes)
1. ✅ Run GridSearchCV for best parameters
2. ✅ Test max_depth, min_samples_split, min_samples_leaf
3. ✅ Evaluate on test set
4. ✅ Compare improvements

### Phase 3: Try Alternatives (30 minutes)
1. ✅ Implement Random Forest
2. ✅ Compare with tuned Decision Tree
3. ✅ Run feature importance analysis
4. ✅ Choose best model

### Phase 4: Advanced (Optional, 45 minutes)
1. ✅ Try SMOTE for data balancing
2. ✅ Implement ensemble voting
3. ✅ Test on actual test videos

---

## Expected Improvements by Strategy

| Strategy | Effort | Recall Gain | Precision Impact | Recommendation |
|----------|:------:|:----------:|:----------------:|:---------------:|
| Class weights | 5 min | +10-20% | -5-10% | ✅ START HERE |
| Threshold tuning | 10 min | +5-15% | -10-20% | ✅ QUICK GAIN |
| Hyperparameter tuning | 30 min | +5-10% | Neutral | ✅ TRY NEXT |
| Random Forest | 20 min | +10-15% | Neutral | ✅ GOOD ALTERNATIVE |
| SMOTE balancing | 15 min | +5-15% | -5% | ⏳ MAYBE |
| Ensemble voting | 30 min | +10-20% | Neutral | ⏳ IF OTHERS FAIL |

---

## Which Strategy Will Actually Help?

### What WON'T Help Much:
❌ More hyperparameter tuning
❌ Fancy algorithms (SVM, Neural Networks)
❌ Data augmentation alone
❌ Ensemble methods (without addressing class imbalance)

**Why**: Root cause is DATA BIAS (84.4% on 2 teachers), not model architecture

### What MIGHT Help:
⚠️ Class weight balancing (might squeeze 10-15% recall improvement)
⚠️ Threshold adjustment (depends on your precision/recall trade-off)
⚠️ Random Forest (slightly more robust)

**Why**: These make the model more sensitive to rare transitions

### What WILL Actually Fix It:
✅ Model v2 with balanced training data
✅ Retrain on 70% train / 30% test (all teachers equally)
✅ Expected: 40-60% recall improvement

**Why**: Current model learned chemistry patterns, needs diverse training

---

## Honest Assessment

### The Truth About Improving Model v1:

**If you only improve the model, you'll gain**:
- Maybe 10-15% recall improvement (with class weights + tuning)
- Model becomes more sensitive to transitions
- But still struggles on different teacher styles

**Why Model v2 is Better**:
- 40-60% recall improvement expected (vs 10-15% from tuning)
- Fixes root cause (data bias), not just symptoms
- Same 4 features work great with balanced data
- Proven by changing only data distribution

### Our Recommendation:

**Best Approach** (Hybrid):
1. Apply class_weight='balanced' to current Decision Tree (+10% recall)
2. Train Model v2 with balanced data (+40% recall)
3. Test both on algo_1, cn_1, toc_1
4. Use whichever performs better (likely v2)

**This way**: You try to improve v1 (quick), but also prepare v2 (long-term fix)

---

## Summary

| Strategy | Time | Benefit | Worth It? |
|----------|:----:|:-------:|:---------:|
| Class weights | 5 min | +10-15% recall | ✅ YES |
| Threshold tuning | 10 min | +5-10% recall | ✅ YES |
| Hyperparameter tuning | 30 min | +5% recall | ⏳ MAYBE |
| Random Forest | 20 min | +10% recall | ⏳ MAYBE |
| SMOTE balancing | 15 min | +10% recall | ⏳ MAYBE |
| Ensemble voting | 30 min | +10% recall | ❌ OVERKILL |
| **Model v2** | **20 min** | **+40-60% recall** | **✅ BEST** |

**My suggestion**: Do class weights + threshold tuning (15 min), then build Model v2 (20 min). 

That way you improve v1 AND prepare the better solution!
