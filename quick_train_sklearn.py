#!/usr/bin/env python3
"""
Quick test of sklearn DecisionTreeClassifier with class balancing.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

print("Loading dataset...")
df = pd.read_csv('labeled_dataset.csv')

# Prepare data
FEATURE_NAMES = ['content_fullness', 'frame_quality', 'is_occluded', 'skin_ratio']

train_df = df[df['split'] == 'train']
test_df = df[df['split'] == 'test']

X_train = train_df[FEATURE_NAMES].values
y_train = train_df['is_transition_gt'].values
X_test = test_df[FEATURE_NAMES].values
y_test = test_df['is_transition_gt'].values

# Normalize
X_min = X_train.min(axis=0)
X_max = X_train.max(axis=0)
X_train_norm = (X_train - X_min) / (X_max - X_min + 1e-8)
X_test_norm = (X_test - X_min) / (X_max - X_min + 1e-8)

print(f"\nTraining data: {len(X_train)} samples")
print(f"  - Positive: {(y_train == 1).sum()} ({(y_train == 1).sum() / len(y_train) * 100:.1f}%)")
print(f"  - Negative: {(y_train == 0).sum()} ({(y_train == 0).sum() / len(y_train) * 100:.1f}%)")

print(f"\nTest data: {len(X_test)} samples")
print(f"  - Positive: {(y_test == 1).sum()} ({(y_test == 1).sum() / len(y_test) * 100:.1f}%)")
print(f"  - Negative: {(y_test == 0).sum()} ({(y_test == 0).sum() / len(y_test) * 100:.1f}%)")

# Train baseline model (no class weighting)
print("\n" + "="*70)
print("BASELINE MODEL (no class weighting)")
print("="*70)
baseline_model = DecisionTreeClassifier(
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
baseline_model.fit(X_train_norm, y_train)
y_pred = baseline_model.predict(X_test_norm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix:")
print(f"  TP: {tp}  FP: {fp}")
print(f"  FN: {fn}  TN: {tn}")
print(f"  Recall: {tp/(tp+fn):.4f}")

# Train balanced model
print("\n" + "="*70)
print("BALANCED MODEL (class_weight='balanced')")
print("="*70)
balanced_model = DecisionTreeClassifier(
    max_depth=15,
    class_weight='balanced',
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
balanced_model.fit(X_train_norm, y_train)
y_pred_balanced = balanced_model.predict(X_test_norm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_balanced, target_names=['Negative', 'Positive']))

cm_balanced = confusion_matrix(y_test, y_pred_balanced)
tn, fp, fn, tp = cm_balanced.ravel()
print(f"\nConfusion Matrix:")
print(f"  TP: {tp}  FP: {fp}")
print(f"  FN: {fn}  TN: {tn}")
print(f"  Recall: {tp/(tp+fn):.4f}")

# Train heavily weighted model
print("\n" + "="*70)
print("HEAVILY WEIGHTED MODEL (1:50 ratio)")
print("="*70)
weighted_model = DecisionTreeClassifier(
    max_depth=15,
    class_weight={0: 1, 1: 50},
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
weighted_model.fit(X_train_norm, y_train)
y_pred_weighted = weighted_model.predict(X_test_norm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_weighted, target_names=['Negative', 'Positive']))

cm_weighted = confusion_matrix(y_test, y_pred_weighted)
tn, fp, fn, tp = cm_weighted.ravel()
print(f"\nConfusion Matrix:")
print(f"  TP: {tp}  FP: {fp}")
print(f"  FN: {fn}  TN: {tn}")
print(f"  Recall: {tp/(tp+fn):.4f}")

# Save best model (balanced)
print("\n" + "="*70)
print("SAVING BALANCED MODEL")
print("="*70)
with open('trained_model_v3_balanced.pkl', 'wb') as f:
    pickle.dump(balanced_model, f)

norm_data = {'X_min': X_min, 'X_max': X_max}
with open('model_v3_normalization.pkl', 'wb') as f:
    pickle.dump(norm_data, f)

print("✅ Saved trained_model_v3_balanced.pkl")
print("✅ Saved model_v3_normalization.pkl")
