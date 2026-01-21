#!/usr/bin/env python3
"""
Select best slide frames around detected transition timestamps.

Process:
- For each detected timestamp, sample frames within a window (±window seconds, step seconds).
- Compute features per frame (content_fullness, occlusion, blur, foreground ratio) and an aHash.
- Cluster by aHash distance to merge duplicates.
- Pick best frame per cluster using a score: content_fullness↑, occlusion↓, blur↓, foreground↓.

Usage:
  python select_best_slides.py \
    --videos data/testing_videos \
    --detections results_postfilter_v3_boost010 \
    --out best_frames \
    --window 2.0 --step 0.2 --hash-thresh 10
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from detect_transitions_universal import FrameFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def ahash(gray: np.ndarray, size: int = 8) -> np.ndarray:
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    mean = resized.mean()
    return (resized > mean).astype(np.uint8)


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


def compute_scores(frame: np.ndarray, extractor: FrameFeatureExtractor, foreground_ratio: float = 0.0) -> Tuple[float, dict]:
    # Reuse extractor to compute content_fullness, occlusion, etc.
    # Edge/motion are not used for scoring here.
    feats = extractor.extract_features(frame, edge_change=0.0, frame_diff_mean=0.0)
    content_fullness = float(feats[0])
    frame_quality = float(feats[1])  # Laplacian-based quality
    is_occluded = float(feats[2])
    skin_ratio = float(feats[3])

    # Score: higher content_fullness and frame_quality, lower occlusion/skin_ratio/foreground
    score = (
        (1.5 * content_fullness)
        + (0.8 * frame_quality)
        - (1.2 * skin_ratio)
        - (0.8 * is_occluded)
        - (1.5 * foreground_ratio)
    )

    return score, {
        "content_fullness": content_fullness,
        "frame_quality": frame_quality,
        "is_occluded": is_occluded,
        "skin_ratio": skin_ratio,
        "foreground_ratio": float(foreground_ratio),
    }


def read_frame_at(cap: cv2.VideoCapture, fps: float, timestamp: float) -> Tuple[np.ndarray, float]:
    frame_idx = max(0, int(round(timestamp * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None, fps
    return frame, fps


def load_detections(det_json: Path) -> List[float]:
    data = json.loads(det_json.read_text())
    if "transitions" in data:
        return [float(t["timestamp"]) for t in data["transitions"]]
    # fallback if format is flat list
    return [float(t) for t in data]


def cluster_and_select(candidates, hash_thresh: int) -> List[dict]:
    clusters = []
    for cand in candidates:
        placed = False
        for cl in clusters:
            if hamming(cand["hash"], cl["hash"]) <= hash_thresh:
                cl["items"].append(cand)
                # keep representative hash (first)
                placed = True
                break
        if not placed:
            clusters.append({"hash": cand["hash"], "items": [cand]})

    selected = []
    for cl in clusters:
        best = max(cl["items"], key=lambda x: x["score"])
        selected.append(best)
    selected.sort(key=lambda x: x["timestamp"])
    return selected


def process_video(
    video_path: Path,
    det_path: Path,
    out_dir: Path,
    window: float,
    step: float,
    hash_thresh: int,
    fg_thresh: float,
    edge_zone: float,
    fg_drop: float,
):
    extractor = FrameFeatureExtractor()
    timestamps = load_detections(det_path)
    if not timestamps:
        logger.warning(f"No detections in {det_path.name}")
        return []

    candidates = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return []
    fps = cap.get(cv2.CAP_PROP_FPS)
    for t in timestamps:
        start = max(0.0, t - window)
        end = t + window
        ts = start
        window_frames = []
        sum_small = None
        count_small = 0
        while ts <= end + 1e-6:
            frame, fps = read_frame_at(cap, fps, ts)
            if frame is None:
                ts += step
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h = ahash(gray)
            small_gray = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
            if sum_small is None:
                sum_small = small_gray.astype(np.float32)
            else:
                sum_small += small_gray.astype(np.float32)
            count_small += 1
            window_frames.append({
                "timestamp": float(ts),
                "small_gray": small_gray,
                "hash": h,
            })
            ts += step

        if not window_frames:
            continue

        if not count_small:
            continue

        # Use mean of downscaled grayscale for foreground estimation
        mean_gray = sum_small / float(count_small)

        for wf in window_frames:
            diff = np.abs(wf["small_gray"].astype(np.float32) - mean_gray) / 255.0
            foreground_ratio = float(np.mean(diff > fg_thresh))
            # estimate foreground center for teacher location
            fg_mask = diff > fg_thresh
            if np.any(fg_mask):
                ys, xs = np.where(fg_mask)
                center_x = float(np.mean(xs)) / fg_mask.shape[1]
            else:
                center_x = 0.5

            # Drop frames with large foreground in the middle (teacher blocking)
            if foreground_ratio >= fg_drop and edge_zone < center_x < (1.0 - edge_zone):
                continue

            frame, _ = read_frame_at(cap, fps, wf["timestamp"])
            if frame is None:
                continue
            score, meta = compute_scores(frame, extractor, foreground_ratio=foreground_ratio)
            candidates.append({
                "timestamp": wf["timestamp"],
                "score": float(score),
                "hash": wf["hash"],
                "meta": meta,
            })

    selected = cluster_and_select(candidates, hash_thresh=hash_thresh)

    # Save selected frames
    saved = []
    video_out = out_dir / video_path.stem
    video_out.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(selected, 1):
        fname = f"{video_path.stem}_slide_{i:03d}_{item['timestamp']:.2f}s.jpg"
        fpath = video_out / fname
        frame, _ = read_frame_at(cap, fps, item["timestamp"])
        if frame is None:
            continue
        cv2.imwrite(str(fpath), frame)
        saved.append({
            "timestamp": item["timestamp"],
            "frame_path": str(fpath),
            "score": item["score"],
            **item["meta"],
        })

    # Save per-video JSON
    out_json = video_out / f"{video_path.stem}_best_frames.json"
    with open(out_json, "w") as f:
        json.dump({"video": video_path.name, "best_frames": saved}, f, indent=2)

    cap.release()

    return saved


def main():
    parser = argparse.ArgumentParser(description="Select best slide frames")
    parser.add_argument("--videos", required=True, help="Video directory")
    parser.add_argument("--detections", required=True, help="Detections directory")
    parser.add_argument("--out", default="best_frames", help="Output directory")
    parser.add_argument("--window", type=float, default=2.0, help="Seconds around detection")
    parser.add_argument("--step", type=float, default=0.2, help="Sampling step seconds")
    parser.add_argument("--hash-thresh", type=int, default=10, help="aHash Hamming distance threshold")
    parser.add_argument("--fg-thresh", type=float, default=0.08, help="Foreground ratio threshold (0-1)")
    parser.add_argument("--edge-zone", type=float, default=0.20, help="Edge zone width (0-0.5) to allow foreground")
    parser.add_argument("--fg-drop", type=float, default=0.18, help="Foreground ratio cutoff to drop middle-teacher frames")
    args = parser.parse_args()

    videos_dir = Path(args.videos)
    det_dir = Path(args.detections)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    det_files = sorted(det_dir.glob("*_detected.json"))
    if not det_files:
        logger.error(f"No *_detected.json files found in {det_dir}")
        return

    summary_rows = []
    for det_path in det_files:
        video_name = det_path.stem.replace("_detected", "") + ".mp4"
        video_path = videos_dir / video_name
        if not video_path.exists():
            logger.warning(f"Video not found: {video_path}")
            continue

        logger.info(f"Processing {video_path.name}")
        saved = process_video(
            video_path,
            det_path,
            out_dir,
            args.window,
            args.step,
            args.hash_thresh,
            args.fg_thresh,
            args.edge_zone,
            args.fg_drop,
        )
        summary_rows.append({
            "video": video_path.name,
            "best_frames": len(saved),
        })

    # Write summary CSV
    summary_csv = out_dir / "best_frames_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "best_frames"])
        writer.writeheader()
        writer.writerows(summary_rows)
    logger.info(f"Summary saved to {summary_csv}")


if __name__ == "__main__":
    main()
