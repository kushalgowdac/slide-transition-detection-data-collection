"""
detect_with_postfilter.py

Apply post-filtering to detected transitions:
- Keep only detections with confidence >= (threshold + confidence_boost)
- Reduces false positives while maintaining recall

Usage:
    python detect_with_postfilter.py data/testing_videos --dir --out results_postfilter \
        --thresh 0.55 --smooth 5 --diff-pct 90 --min-gap 3.0 \
        --confidence-boost 0.10 \
        --model trained_model_gb_enriched_v3.pkl --norm model_gb_enriched_v3_normalization.pkl
"""

import argparse
import numpy as np
import json
import pickle
from pathlib import Path
import sys

# Import existing detector functions
from detect_transitions_universal import FrameFeatureExtractor, load_model_and_normalization


def extract_all_frames_with_probs(video_path, model, X_min, X_max):
    """
    Extract features for all frames and run model predictions.
    Uses existing FrameFeatureExtractor.
    Returns: times, probs, features
    """
    extractor = FrameFeatureExtractor()
    features, fps, duration, diff_scores = extractor.extract_all_frames(video_path)
    
    if features is None:
        return None, None, None
    
    # Normalize features
    features_norm, _, _ = extractor.normalize_features(features, X_min, X_max)
    
    # Get probabilities
    probs = model.predict_proba(features_norm)[:, 1]
    
    # Calculate timestamps
    times = np.arange(len(features)) / fps
    
    return times, probs, features


def detect_transitions_with_postfilter(
    video_path,
    model,
    X_min,
    X_max,
    threshold=0.5,
    smooth_window=5,
    diff_pct=90,
    min_gap=2.0,
    confidence_boost=0.10
):
    """
    Detect transitions with post-filtering based on confidence cutoff.
    
    Args:
        video_path: Path to video
        model: Trained classifier
        X_min, X_max: Normalization params
        threshold: Base probability threshold
        smooth_window: Smoothing window size
        diff_pct: Percentile for frame-diff gating
        min_gap: Minimum gap between transitions (seconds)
        confidence_boost: Additional confidence requirement (e.g., 0.10)
    
    Returns:
        List of (timestamp, confidence) tuples
    """
    times, probs, features = extract_all_frames_with_probs(video_path, model, X_min, X_max)
    
    if times is None:
        return []
    
    # Step 1: Apply base threshold
    candidates = probs >= threshold
    
    # Step 2: Smoothing
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        probs_smooth = np.convolve(probs, kernel, mode='same')
        candidates = probs_smooth >= threshold
    else:
        probs_smooth = probs
    
    # Step 3: Frame-diff gating
    if diff_pct > 0 and len(features) > 1:
        frame_diff_mean = features[:, 5]  # 6th feature
        cutoff = np.percentile(frame_diff_mean, diff_pct)
        high_diff = frame_diff_mean >= cutoff
        candidates = candidates & high_diff
    
    # Step 4: Temporal clustering
    detections = []
    cluster_start = None
    cluster_probs = []
    cluster_times = []
    
    for i, is_cand in enumerate(candidates):
        if is_cand:
            if cluster_start is None:
                cluster_start = i
            cluster_probs.append(probs_smooth[i])
            cluster_times.append(times[i])
        else:
            if cluster_start is not None:
                # End of cluster - pick max confidence
                max_idx = np.argmax(cluster_probs)
                peak_time = cluster_times[max_idx]
                peak_conf = cluster_probs[max_idx]
                detections.append((peak_time, peak_conf))
                
                cluster_start = None
                cluster_probs = []
                cluster_times = []
    
    # Handle last cluster
    if cluster_start is not None:
        max_idx = np.argmax(cluster_probs)
        peak_time = cluster_times[max_idx]
        peak_conf = cluster_probs[max_idx]
        detections.append((peak_time, peak_conf))
    
    # Step 5: Merge close detections
    if min_gap > 0 and len(detections) > 1:
        merged = []
        current = detections[0]
        
        for next_det in detections[1:]:
            if next_det[0] - current[0] < min_gap:
                # Keep higher confidence
                if next_det[1] > current[1]:
                    current = next_det
            else:
                merged.append(current)
                current = next_det
        
        merged.append(current)
        detections = merged
    
    # Step 6: POST-FILTER by confidence cutoff
    confidence_cutoff = threshold + confidence_boost
    filtered = [(t, c) for t, c in detections if c >= confidence_cutoff]
    
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Detect transitions with post-filtering")
    parser.add_argument("video_path", help="Video file or directory")
    parser.add_argument("--dir", action="store_true", help="Process directory")
    parser.add_argument("--out", default="results_postfilter", help="Output directory")
    parser.add_argument("--model", default="trained_model_gb_enriched_v2.pkl")
    parser.add_argument("--norm", default="model_gb_enriched_v2_normalization.pkl")
    parser.add_argument("--thresh", type=float, default=0.5, help="Base threshold")
    parser.add_argument("--smooth", type=int, default=5, help="Smoothing window")
    parser.add_argument("--diff-pct", type=float, default=90, help="Diff percentile")
    parser.add_argument("--min-gap", type=float, default=2.0, help="Min gap (seconds)")
    parser.add_argument("--confidence-boost", type=float, default=0.10, 
                        help="Additional confidence requirement above threshold")
    
    args = parser.parse_args()
    
    # Load model
    model, X_min, X_max = load_model_and_normalization(args.model, args.norm)
    
    # Create output dir
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video files
    if args.dir:
        video_files = sorted(Path(args.video_path).glob("*.mp4"))
    else:
        video_files = [Path(args.video_path)]
    
    print(f"Processing {len(video_files)} videos...")
    print(f"Base threshold: {args.thresh}")
    print(f"Confidence cutoff: {args.thresh + args.confidence_boost}")
    print(f"Smooth window: {args.smooth}, Diff percentile: {args.diff_pct}, Min gap: {args.min_gap}s")
    print()
    
    for video_path in video_files:
        print(f"Processing: {video_path.name}")
        
        detections = detect_transitions_with_postfilter(
            video_path,
            model,
            X_min,
            X_max,
            threshold=args.thresh,
            smooth_window=args.smooth,
            diff_pct=args.diff_pct,
            min_gap=args.min_gap,
            confidence_boost=args.confidence_boost
        )
        
        print(f"  Detected {len(detections)} transitions (after post-filter)")
        
        # Save results
        video_name = video_path.stem
        
        # JSON
        json_out = out_dir / f"{video_name}.json"
        json_data = {
            "video": video_path.name,
            "num_transitions": len(detections),
            "transitions": [
                {"timestamp": float(t), "confidence": float(c)}
                for t, c in detections
            ],
            "params": {
                "threshold": args.thresh,
                "smooth_window": args.smooth,
                "diff_pct": args.diff_pct,
                "min_gap": args.min_gap,
                "confidence_boost": args.confidence_boost,
                "confidence_cutoff": args.thresh + args.confidence_boost
            }
        }
        with open(json_out, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # TXT
        txt_out = out_dir / f"{video_name}.txt"
        with open(txt_out, 'w') as f:
            for t, c in detections:
                f.write(f"{t:.2f}\n")
        
        print(f"  Saved: {json_out.name}, {txt_out.name}")
    
    print(f"\nAll results saved to: {out_dir}")


if __name__ == "__main__":
    main()
