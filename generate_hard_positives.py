#!/usr/bin/env python3
"""
Generate hard positive samples from given videos by sampling frames within a
window around known transition times. Outputs a CSV compatible with training.

Example:
  & "D:/College_Life/projects/slide transition detection - data collection/.venv/Scripts/python.exe" \
      generate_hard_positives.py \
      --videos data/testing_videos \
      --out hard_positives.csv \
      --window 1.0 \
      --step 0.2
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import List, Set

import numpy as np

from compare_with_ground_truth import find_ground_truth_file
from detect_transitions_universal import FrameFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_ground_truth_seconds(video_stem: str) -> List[float]:
    gt_path = find_ground_truth_file(video_stem, [
        "data/testing_videos",
        "data/ground_truth",
    ])
    if not gt_path:
        logger.warning(f"No ground truth for {video_stem}; using empty list")
        return []
    times: List[float] = []
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                times.append(float(line))
            except ValueError:
                continue
    return times


def collect_indices(gt_times: List[float], fps: float, total_frames: int, window: float, step: float) -> Set[int]:
    indices: Set[int] = set()
    half = window
    stride = max(1, int(fps * step))
    for t in gt_times:
        start = max(0, int((t - half) * fps))
        end = min(total_frames - 1, int((t + half) * fps))
        for idx in range(start, end + 1, stride):
            indices.add(idx)
    return indices


def main():
    parser = argparse.ArgumentParser(description="Generate hard positive frames")
    parser.add_argument("--videos", default="data/testing_videos", help="Directory with videos")
    parser.add_argument("--out", default="hard_positives.csv", help="Output CSV path")
    parser.add_argument("--window", type=float, default=1.0, help="Seconds around GT transitions")
    parser.add_argument("--step", type=float, default=0.2, help="Seconds between sampled frames within window")
    args = parser.parse_args()

    video_dir = Path(args.videos)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    video_files = sorted(video_dir.glob("*.mp4"))
    if not video_files:
        logger.error(f"No videos found in {video_dir}")
        return

    rows = []
    for video_path in video_files:
        logger.info("\n" + "=" * 70)
        logger.info(f"VIDEO: {video_path.name}")
        gt_times = load_ground_truth_seconds(video_path.stem)
        if not gt_times:
            logger.warning(f"No transitions for {video_path.name}; skipping")
            continue

        extractor = FrameFeatureExtractor()
        features, fps, duration, diff_scores = extractor.extract_all_frames(str(video_path))
        if features is None:
            continue

        idx_set = collect_indices(gt_times, fps, len(features), args.window, args.step)
        for idx in sorted(idx_set):
            ts = idx / fps
            feats = features[idx]
            rows.append({
                "video_name": video_path.name,
                "frame_idx": idx,
                "timestamp": ts,
                "source_video": str(video_path),
                "video_id": video_path.stem,
                "content_fullness": feats[0],
                "frame_quality": feats[1],
                "is_occluded": feats[2],
                "skin_ratio": feats[3],
                "edge_change": feats[4] if len(feats) > 4 else 0.0,
                "frame_diff_mean": feats[5] if len(feats) > 5 else 0.0,
                "is_transition_gt": 1,
                "split": "train",
            })

        logger.info(f"  Collected {len([r for r in rows if r['video_name']==video_path.name])} positives so far for this video")

    if not rows:
        logger.warning("No rows collected; nothing to write")
        return

    fieldnames = [
        "video_name",
        "frame_idx",
        "timestamp",
        "source_video",
        "video_id",
        "content_fullness",
        "frame_quality",
        "is_occluded",
        "skin_ratio",
        "edge_change",
        "frame_diff_mean",
        "is_transition_gt",
        "split",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Wrote {len(rows)} hard positives to {out_path}")


if __name__ == "__main__":
    main()
