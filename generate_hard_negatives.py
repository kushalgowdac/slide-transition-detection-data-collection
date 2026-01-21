#!/usr/bin/env python3
"""
Generate hard negative samples from given videos by sampling frames far from known
transitions. Outputs a CSV compatible with training scripts.

Example:
  & "D:/College_Life/projects/slide transition detection - data collection/.venv/Scripts/python.exe" \
      generate_hard_negatives.py \
      --videos data/testing_videos \
      --out hard_negatives.csv \
      --sample-rate 3.0 \
      --exclude-window 12.0
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import List

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


def should_keep(timestamp: float, gt_times: List[float], exclude_window: float) -> bool:
    if not gt_times:
        return True
    return all(abs(timestamp - t) >= exclude_window for t in gt_times)


def main():
    parser = argparse.ArgumentParser(description="Generate hard negative frames")
    parser.add_argument("--videos", default="data/testing_videos", help="Directory with videos")
    parser.add_argument("--out", default="hard_negatives.csv", help="Output CSV path")
    parser.add_argument("--sample-rate", type=float, default=3.0, help="Seconds between sampled frames")
    parser.add_argument("--exclude-window", type=float, default=12.0, help="Seconds to exclude around GT transitions")
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

        extractor = FrameFeatureExtractor()
        features, fps, duration, diff_scores = extractor.extract_all_frames(str(video_path))
        if features is None:
            continue

        stride = max(1, int(fps * args.sample_rate))
        for idx in range(0, len(features), stride):
            ts = idx / fps
            if not should_keep(ts, gt_times, args.exclude_window):
                continue
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
                "is_transition_gt": 0,
                "split": "train",
            })

        logger.info(f"  Collected {len([r for r in rows if r['video_name']==video_path.name])} negatives so far for this video")

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

    logger.info(f"Wrote {len(rows)} hard negatives to {out_path}")


if __name__ == "__main__":
    main()
