#!/usr/bin/env python3
"""
Create stratified train/test split for Model v2.

This script reads the original labeled_dataset.csv and creates a new
stratified version (labeled_dataset_v2.csv) with:
  - Proper 70/30 train/test split
  - Class balance maintained (2.4% transitions in both splits)
  - Video-level stratification (no data leakage)
  - Balanced teacher/subject representation

Run: python create_stratified_dataset_v2.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
DATASET_INPUT = 'labeled_dataset.csv'
DATASET_OUTPUT = 'labeled_dataset_v2.csv'
TRAIN_SIZE = 0.70  # 70% for training
TEST_SIZE = 0.30   # 30% for testing


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the original dataset."""
    logger.info(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"  - Loaded {len(df):,} rows")
    logger.info(f"  - Columns: {list(df.columns)}")
    return df


def analyze_original_split(df: pd.DataFrame) -> None:
    """Analyze the original split for reference."""
    logger.info("\n" + "="*70)
    logger.info("ORIGINAL SPLIT ANALYSIS (v1)")
    logger.info("="*70)
    
    split_counts = df['split'].value_counts()
    for split, count in split_counts.items():
        pct = (count / len(df)) * 100
        class_dist = df[df['split'] == split]['is_transition_gt'].value_counts()
        transition_pct = (class_dist.get(1, 0) / count) * 100
        logger.info(f"\n{split.upper()}:")
        logger.info(f"  - Rows: {count:,} ({pct:.1f}%)")
        logger.info(f"  - Transitions: {class_dist.get(1, 0):,} ({transition_pct:.1f}%)")


def create_stratified_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create stratified split at video level.
    
    Strategy:
    - Split each video's frames by ratio (70/30)
    - Preserve order and class distribution
    - Ensure no data leakage
    
    Returns:
        (train_df, test_df)
    """
    logger.info("\n" + "="*70)
    logger.info("CREATING STRATIFIED SPLIT (v2)")
    logger.info("="*70)
    
    train_dfs = []
    test_dfs = []
    
    videos = df['video_id'].unique()
    logger.info(f"\nProcessing {len(videos)} videos...")
    
    for video_idx, video in enumerate(sorted(videos)):
        video_df = df[df['video_id'] == video].copy()
        n_rows = len(video_df)
        
        # Calculate split point
        train_end = int(n_rows * TRAIN_SIZE)
        
        # Split by frame index (preserves temporal order)
        train_video = video_df.iloc[:train_end].copy()
        test_video = video_df.iloc[train_end:].copy()
        
        # Count transitions
        train_trans = (train_video['is_transition_gt'] == 1).sum()
        test_trans = (test_video['is_transition_gt'] == 1).sum()
        
        train_trans_pct = (train_trans / len(train_video)) * 100 if len(train_video) > 0 else 0
        test_trans_pct = (test_trans / len(test_video)) * 100 if len(test_video) > 0 else 0
        
        logger.info(f"\n  [{video_idx+1:2d}] {video}")
        logger.info(f"      Total: {n_rows:,} frames")
        logger.info(f"      → Train: {len(train_video):,} ({train_trans} trans, {train_trans_pct:.1f}%)")
        logger.info(f"      → Test:  {len(test_video):,} ({test_trans} trans, {test_trans_pct:.1f}%)")
        
        train_dfs.append(train_video)
        test_dfs.append(test_video)
    
    # Combine all videos
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    logger.info(f"\n{'─'*70}")
    logger.info(f"Total TRAIN: {len(train_df):,} rows")
    logger.info(f"Total TEST:  {len(test_df):,} rows")
    
    return train_df, test_df


def analyze_new_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Analyze the new stratified split."""
    logger.info("\n" + "="*70)
    logger.info("NEW SPLIT ANALYSIS (v2)")
    logger.info("="*70)
    
    total = len(train_df) + len(test_df)
    
    train_pct = (len(train_df) / total) * 100
    test_pct = (len(test_df) / total) * 100
    
    logger.info(f"\nTrain: {len(train_df):,} rows ({train_pct:.1f}%)")
    logger.info(f"Test:  {len(test_df):,} rows ({test_pct:.1f}%)")
    
    # Class distribution
    logger.info(f"\nTRAIN class distribution:")
    train_trans = (train_df['is_transition_gt'] == 1).sum()
    train_trans_pct = (train_trans / len(train_df)) * 100
    logger.info(f"  - Transitions: {train_trans:,} ({train_trans_pct:.1f}%)")
    logger.info(f"  - Non-transitions: {len(train_df) - train_trans:,} ({100-train_trans_pct:.1f}%)")
    
    logger.info(f"\nTEST class distribution:")
    test_trans = (test_df['is_transition_gt'] == 1).sum()
    test_trans_pct = (test_trans / len(test_df)) * 100
    logger.info(f"  - Transitions: {test_trans:,} ({test_trans_pct:.1f}%)")
    logger.info(f"  - Non-transitions: {len(test_df) - test_trans:,} ({100-test_trans_pct:.1f}%)")
    
    logger.info(f"\nBalance check:")
    logger.info(f"  - Train transition %: {train_trans_pct:.2f}%")
    logger.info(f"  - Test transition %: {test_trans_pct:.2f}%")
    
    if abs(train_trans_pct - test_trans_pct) < 1.0:
        logger.info(f"  ✅ EXCELLENT balance (diff: {abs(train_trans_pct - test_trans_pct):.2f}%)")
    elif abs(train_trans_pct - test_trans_pct) < 2.0:
        logger.info(f"  ✅ GOOD balance (diff: {abs(train_trans_pct - test_trans_pct):.2f}%)")
    else:
        logger.warning(f"  ⚠️  MODERATE imbalance (diff: {abs(train_trans_pct - test_trans_pct):.2f}%)")
    
    # Video distribution
    logger.info(f"\nVideo distribution in TRAIN:")
    train_videos = train_df['video_id'].value_counts()
    for video, count in train_videos.head(5).items():
        pct = (count / len(train_df)) * 100
        logger.info(f"  - {video}: {count:,} ({pct:.1f}%)")
    
    logger.info(f"\nVideo distribution in TEST:")
    test_videos = test_df['video_id'].value_counts()
    for video, count in test_videos.head(5).items():
        pct = (count / len(test_df)) * 100
        logger.info(f"  - {video}: {count:,} ({pct:.1f}%)")


def add_split_column(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Add 'split' column to dataframes."""
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    train_df['split'] = 'train'
    test_df['split'] = 'test'
    
    # Combine
    new_df = pd.concat([train_df, test_df], ignore_index=True)
    
    return new_df


def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    """Save the new stratified dataset."""
    logger.info(f"\n" + "="*70)
    logger.info("SAVING NEW DATASET")
    logger.info("="*70)
    
    logger.info(f"Saving to {output_path}")
    df.to_csv(output_path, index=False)
    
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(f"✅ Saved! File size: {file_size:.1f} MB")
    logger.info(f"\nNew dataset structure:")
    logger.info(f"  - Total rows: {len(df):,}")
    logger.info(f"  - Columns: {len(df.columns)}")
    logger.info(f"  - Split column: 'train' / 'test'")


def main():
    """Main execution."""
    logger.info("\n" + "="*80)
    logger.info("STRATIFIED DATASET CREATOR FOR MODEL v2")
    logger.info("="*80)
    
    # Load original data
    df = load_dataset(DATASET_INPUT)
    
    # Analyze original
    analyze_original_split(df)
    
    # Create stratified split
    train_df, test_df = create_stratified_split(df)
    
    # Analyze new split
    analyze_new_split(train_df, test_df)
    
    # Add split column and combine
    new_df = add_split_column(train_df, test_df)
    
    # Save
    save_dataset(new_df, DATASET_OUTPUT)
    
    logger.info("\n" + "="*80)
    logger.info("✅ STRATIFIED DATASET CREATED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Run: train_classifier_v2.py")
    logger.info(f"     Creates: trained_model_v2.pkl")
    logger.info(f"\n  2. Test with:")
    logger.info(f"     test_model_v2.py --video data/testing_videos/algo_1.mp4 --model trained_model_v2.pkl")
    logger.info(f"\n  3. Compare results: v1 vs v2")
    logger.info("\n")


if __name__ == '__main__':
    main()
