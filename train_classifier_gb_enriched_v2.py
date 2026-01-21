#!/usr/bin/env python3
"""
Train an enriched model with edge and motion features, optionally adding hard negatives.
Supports GradientBoosting (default) or RandomForest with class balancing.

Example:
  & "D:/College_Life/projects/slide transition detection - data collection/.venv/Scripts/python.exe" \
      train_classifier_gb_enriched_v2.py \
      --dataset labeled_dataset.csv \
      --extra-negatives hard_negatives.csv \
      --model-out trained_model_gb_enriched_v2.pkl \
      --norm-out model_gb_enriched_v2_normalization.pkl
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

DEFAULT_FEATURES = [
    "content_fullness",
    "frame_quality",
    "is_occluded",
    "skin_ratio",
    "edge_change",
    "frame_diff_mean",
]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_dataset(csv_path: str, required_features: List[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in required_features if c not in df.columns]
    if missing:
        logger.warning(f"Columns missing in {csv_path}: {missing} -> filling with 0")
        for c in missing:
            df[c] = 0.0
    return df


def prepare_split(df: pd.DataFrame, split: str, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    split_df = df[df['split'] == split]
    X = split_df[feature_names].values
    y = split_df['is_transition_gt'].values
    pos = (y == 1).sum()
    logger.info(
        f"{split.upper()} set: {len(X):,} samples | positives: {pos:,} ({(pos/len(y))*100:.2f}%)"
    )
    return X, y


def normalize_train(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_norm = (X_train - X_min) / (X_max - X_min + 1e-8)
    logger.info(f"Feature mins: {X_min}")
    logger.info(f"Feature maxs: {X_max}")
    return X_norm, X_min, X_max


def normalize_test(X: np.ndarray, X_min: np.ndarray, X_max: np.ndarray) -> np.ndarray:
    return (X - X_min) / (X_max - X_min + 1e-8)


def compute_sample_weights(y: np.ndarray, pos_boost: float = 1.0) -> np.ndarray:
    pos = (y == 1).sum()
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return np.ones_like(y, dtype=float)
    pos_weight = (neg / max(pos, 1)) * pos_boost
    weights = np.where(y == 1, pos_weight, 1.0)
    logger.info(f"Class weights -> pos: {pos_weight:.2f}, neg: 1.0")
    return weights


def train_gb(X_train: np.ndarray, y_train: np.ndarray, sample_weight: np.ndarray) -> GradientBoostingClassifier:
    logger.info("Training GradientBoostingClassifier (deeper trees, more estimators)...")
    model = GradientBoostingClassifier(
        n_estimators=600,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.9,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def train_rf(X_train: np.ndarray, y_train: np.ndarray, sample_weight: np.ndarray) -> RandomForestClassifier:
    logger.info("Training RandomForestClassifier (balanced_subsample)...")
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=16,
        min_samples_leaf=2,
        min_samples_split=4,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=42,
        max_features='sqrt'
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def evaluate(model, X, y, label: str):
    y_pred = model.predict(X)
    logger.info(f"\nEvaluation - {label}")
    logger.info(classification_report(y, y_pred, target_names=['neg', 'pos']))
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    logger.info(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
    logger.info(f"Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}")


def save_artifacts(model, X_min, X_max, model_out: str, norm_out: str):
    with open(model_out, 'wb') as f:
        pickle.dump({'model': model}, f)
    with open(norm_out, 'wb') as f:
        pickle.dump({'X_min': X_min, 'X_max': X_max}, f)
    logger.info(f"Saved model to {model_out}")
    logger.info(f"Saved normalization to {norm_out}")


def main():
    parser = argparse.ArgumentParser(description="Train enriched transition model v2")
    parser.add_argument("--dataset", default="labeled_dataset.csv", help="Base labeled dataset")
    parser.add_argument("--extra-negatives", default=None, help="Optional CSV of hard negatives")
    parser.add_argument("--extra-positives", default=None, help="Optional CSV of hard positives")
    parser.add_argument("--model-out", default="trained_model_gb_enriched_v2.pkl", help="Output model pickle")
    parser.add_argument("--norm-out", default="model_gb_enriched_v2_normalization.pkl", help="Output normalization pickle")
    parser.add_argument("--pos-boost", type=float, default=1.2, help="Multiplier on positive class weight")
    parser.add_argument("--use-rf", action='store_true', help="Use RandomForest instead of GradientBoosting")
    args = parser.parse_args()

    feature_names = DEFAULT_FEATURES

    if not Path(args.dataset).exists():
        logger.error(f"Dataset not found: {args.dataset}")
        return

    df = load_dataset(args.dataset, feature_names)

    if args.extra_negatives:
        neg_path = Path(args.extra_negatives)
        if not neg_path.exists():
            logger.error(f"Extra negatives file not found: {neg_path}")
            return
        df_neg = load_dataset(str(neg_path), feature_names)
        before = len(df)
        df = pd.concat([df, df_neg], ignore_index=True)
        logger.info(f"Added {len(df_neg):,} hard negatives (total rows: {len(df):,}, prev {before:,})")

    if args.extra_positives:
        pos_path = Path(args.extra_positives)
        if not pos_path.exists():
            logger.error(f"Extra positives file not found: {pos_path}")
            return
        df_pos = load_dataset(str(pos_path), feature_names)
        before = len(df)
        df = pd.concat([df, df_pos], ignore_index=True)
        logger.info(f"Added {len(df_pos):,} hard positives (total rows: {len(df):,}, prev {before:,})")

    X_train, y_train = prepare_split(df, 'train', feature_names)
    X_test, y_test = prepare_split(df, 'test', feature_names)

    X_train_norm, X_min, X_max = normalize_train(X_train)
    X_test_norm = normalize_test(X_test, X_min, X_max)

    weights = compute_sample_weights(y_train, pos_boost=args.pos_boost)

    if args.use_rf:
        model = train_rf(X_train_norm, y_train, weights)
    else:
        model = train_gb(X_train_norm, y_train, weights)

    evaluate(model, X_train_norm, y_train, 'TRAIN')
    evaluate(model, X_test_norm, y_test, 'TEST')

    save_artifacts(model, X_min, X_max, args.model_out, args.norm_out)


if __name__ == '__main__':
    main()
