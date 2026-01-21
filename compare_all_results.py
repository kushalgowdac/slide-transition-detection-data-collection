"""
Batch-compare all detected results with available ground truth files.
Outputs a summary table and per-video comparison JSONs.
"""

import json
import logging
from pathlib import Path
from typing import List

from compare_with_ground_truth import TransitionComparator, find_ground_truth_file

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def compare_directory(results_dir: str, tolerance: float = 0.5, summary_out: str = None) -> List[dict]:
    results_path = Path(results_dir)
    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return []

    detected_files = sorted(results_path.glob('*_detected.json'))
    if not detected_files:
        logger.error(f"No detected JSON files found in: {results_dir}")
        return []

    logger.info(f"Found {len(detected_files)} detected file(s) to compare\n")

    all_rows = []
    for i, det_file in enumerate(detected_files, 1):
        video_name = det_file.stem.replace('_detected', '')
        logger.info(f"[{i}/{len(detected_files)}] {video_name}")

        gt_path = find_ground_truth_file(video_name, [
            'data/testing_videos',
            'data/ground_truth',
        ])

        if not gt_path:
            logger.warning(f"  No ground truth found for {video_name}; skipping comparison")
            continue

        comp = TransitionComparator(tolerance_seconds=tolerance)
        res = comp.compare(gt_path, str(det_file))
        if not res:
            continue

        row = {
            'video': video_name,
            'ground_truth': res['ground_truth_count'],
            'detected': res['detected_count'],
            'tp': res['true_positives'],
            'fp': res['false_positives'],
            'fn': res['false_negatives'],
            'precision': res['metrics']['precision'],
            'recall': res['metrics']['recall'],
            'f1': res['metrics']['f1'],
        }
        all_rows.append(row)

        # Save per-video comparison JSON next to detected file
        out_json = det_file.with_name(f"{video_name}_comparison.json")
        with open(out_json, 'w') as f:
            json.dump(res, f, indent=2)
        logger.info(f"  Saved comparison: {out_json}\n")

    # Optional summary CSV
    if summary_out and all_rows:
        import csv
        with open(summary_out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            for r in all_rows:
                writer.writerow(r)
        logger.info(f"Summary CSV saved: {summary_out}")

    # Print compact table
    if all_rows:
        logger.info("\nSUMMARY TABLE (precision/recall/f1)")
        for r in all_rows:
            logger.info(
                f"  {r['video']:<20} GT:{r['ground_truth']:>3}  DET:{r['detected']:>3}  "
                f"TP:{r['tp']:>3}  FP:{r['fp']:>3}  FN:{r['fn']:>3}  "
                f"P:{r['precision']:.2f}  R:{r['recall']:.2f}  F1:{r['f1']:.2f}"
            )

    return all_rows


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python compare_all_results.py <results_dir> [tolerance] [summary_csv]")
        print("\nExample:")
        print("  python compare_all_results.py results/ 0.5 results/summary.csv")
        sys.exit(1)

    res_dir = sys.argv[1]
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    summary = sys.argv[3] if len(sys.argv) > 3 else None

    compare_directory(res_dir, tol, summary)
