#!/usr/bin/env python3
"""
Parameter sweep for transition detection post-processing.
Runs detect_transitions_universal over a grid of (threshold, diff_pct, min_gap)
using the enriched model by default, compares to ground truth, and writes a
summary CSV. Features are extracted once per video and reused across configs.

Example:
  & "D:/College_Life/projects/slide transition detection - data collection/.venv/Scripts/python.exe" sweep_params.py \
        --videos data/testing_videos \
        --out results_sweep \
        --thresholds 0.28,0.32,0.36 \
        --diff-pcts 80,85 \
        --min-gaps 2.0,2.5 \
        --tolerance 10
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from compare_with_ground_truth import TransitionComparator, find_ground_truth_file
from detect_transitions_universal import (
    FrameFeatureExtractor,
    detect_transitions,
    load_model_and_normalization,
    save_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_list_floats(raw: str) -> List[float]:
    return [float(x) for x in raw.split(",") if x]


def config_key(thresh: float, diff_pct: float, min_gap: float) -> str:
    return f"t{thresh:.2f}_d{diff_pct:.0f}_g{min_gap:.1f}"


def load_or_extract(video_path: Path, cache_dir: Path | None) -> Tuple[np.ndarray, float, float, np.ndarray]:
    cache_path = None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{video_path.stem}.npz"
        if cache_path.exists():
            data = np.load(cache_path)
            fps_val = float(data["fps"][0]) if data["fps"].shape else float(data["fps"])
            dur_val = float(data["duration"][0]) if data["duration"].shape else float(data["duration"])
            return data["features"], fps_val, dur_val, data["diff_scores"]

    extractor = FrameFeatureExtractor()
    features, fps, duration, diff_scores = extractor.extract_all_frames(str(video_path))
    if cache_path:
        np.savez_compressed(
            cache_path,
            features=features,
            fps=np.array([fps], dtype=np.float32),
            duration=np.array([duration], dtype=np.float32),
            diff_scores=diff_scores,
        )
    return features, fps, duration, diff_scores


def run_sweep(
    video_dir: Path,
    out_root: Path,
    thresholds: List[float],
    diff_pcts: List[float],
    min_gaps: List[float],
    tolerance: float,
    model_path: str | None,
    norm_path: str | None,
    cache_dir: Path | None,
) -> List[Dict]:
    out_root.mkdir(parents=True, exist_ok=True)

    model, X_min, X_max = load_model_and_normalization(
        model_path or "trained_model_gb_enriched.pkl",
        norm_path or "model_gb_enriched_normalization.pkl",
    )
    if model is None:
        raise SystemExit("Model load failed")

    video_files = [f for f in video_dir.glob("*.mp4")]
    if not video_files:
        raise SystemExit(f"No mp4 files found in {video_dir}")

    summary_rows: List[Dict] = []

    for video_path in video_files:
        logger.info("\n" + "=" * 70)
        logger.info(f"VIDEO: {video_path.name}")
        features, fps, duration, diff_scores = load_or_extract(video_path, cache_dir)
        if features is None:
            continue

        gt_path = find_ground_truth_file(video_path.stem, [
            "data/testing_videos",
            "data/ground_truth",
        ])
        if not gt_path:
            logger.warning(f"No ground truth found for {video_path.stem}; skipping comparisons")

        for thresh in thresholds:
            for diff_pct in diff_pcts:
                for min_gap in min_gaps:
                    tag = config_key(thresh, diff_pct, min_gap)
                    out_dir = out_root / tag
                    out_dir.mkdir(parents=True, exist_ok=True)

                    transitions = detect_transitions(
                        features,
                        fps,
                        model,
                        X_min,
                        X_max,
                        confidence_threshold=thresh,
                        smooth_window=5,
                        diff_scores=diff_scores,
                        diff_percentile=diff_pct,
                        min_gap_seconds=min_gap,
                    )

                    res_paths = save_results(str(video_path), transitions, fps, duration, out_dir)
                    json_path = res_paths[0]

                    if gt_path:
                        comp = TransitionComparator(tolerance_seconds=tolerance)
                        res = comp.compare(gt_path, str(json_path))
                        if res:
                            summary_rows.append({
                                "config": tag,
                                "threshold": thresh,
                                "diff_pct": diff_pct,
                                "min_gap": min_gap,
                                "video": video_path.stem,
                                "ground_truth": res["ground_truth_count"],
                                "detected": res["detected_count"],
                                "tp": res["true_positives"],
                                "fp": res["false_positives"],
                                "fn": res["false_negatives"],
                                "precision": res["metrics"]["precision"],
                                "recall": res["metrics"]["recall"],
                                "f1": res["metrics"]["f1"],
                            })

    if summary_rows:
        summary_path = out_root / "summary_sweep.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "config",
                    "threshold",
                    "diff_pct",
                    "min_gap",
                    "video",
                    "ground_truth",
                    "detected",
                    "tp",
                    "fp",
                    "fn",
                    "precision",
                    "recall",
                    "f1",
                ],
            )
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        logger.info(f"Sweep summary saved: {summary_path}")

        # Print top configs by F1 (average across videos)
        agg: Dict[str, List[float]] = {}
        for row in summary_rows:
            agg.setdefault(row["config"], []).append(row["f1"])
        best = sorted([(np.mean(v), k) for k, v in agg.items()], reverse=True)
        logger.info("Top configs by mean F1:")
        for mean_f1, key in best[:5]:
            logger.info(f"  {key} -> mean F1={mean_f1:.3f}")

    return summary_rows


def main():
    parser = argparse.ArgumentParser(description="Sweep detection thresholds/diff/min-gap")
    parser.add_argument("--videos", default="data/testing_videos", help="Directory with test videos")
    parser.add_argument("--out", default="results_sweep", help="Output root directory")
    parser.add_argument("--thresholds", default="0.28,0.32,0.36", help="Comma-separated confidence thresholds")
    parser.add_argument("--diff-pcts", default="80,85", help="Comma-separated diff percentiles")
    parser.add_argument("--min-gaps", default="2.0,2.5", help="Comma-separated min-gap seconds")
    parser.add_argument("--tolerance", type=float, default=10.0, help="Tolerance for comparison in seconds")
    parser.add_argument("--model", default=None, help="Model pickle path")
    parser.add_argument("--norm", default=None, help="Normalization pickle path")
    parser.add_argument("--cache-dir", default="feature_cache", help="Directory to cache extracted features")
    args = parser.parse_args()

    thresholds = parse_list_floats(args.thresholds)
    diff_pcts = parse_list_floats(args.diff_pcts)
    min_gaps = parse_list_floats(args.min_gaps)

    run_sweep(
        video_dir=Path(args.videos),
        out_root=Path(args.out),
        thresholds=thresholds,
        diff_pcts=diff_pcts,
        min_gaps=min_gaps,
        tolerance=args.tolerance,
        model_path=args.model,
        norm_path=args.norm,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )


if __name__ == "__main__":
    main()
