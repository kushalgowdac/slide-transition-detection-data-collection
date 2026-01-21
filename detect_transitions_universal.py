"""
Universal Transition Detection Script
Works with ANY video file (no ground truth required)
Saves detected transitions with timestamps and confidence scores
"""

import cv2
import pickle
import numpy as np
import json
import logging
from pathlib import Path
from sklearn.cluster import DBSCAN
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class FrameFeatureExtractor:
    """Extract features from video frames for transition detection"""
    
    def __init__(self):
        self.features = []
        self.frames_data = []
        
    def extract_features(self, frame, edge_change=0.0, frame_diff_mean=0.0):
        """Extract color/content cues plus edge/motion signals."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            white_mask = cv2.inRange(gray, 240, 255)
            content_fullness = 1.0 - (cv2.countNonZero(white_mask) / white_mask.size)
            
            frame_quality = cv2.Laplacian(gray, cv2.CV_64F).var()
            frame_quality = min(1.0, frame_quality / 100.0)
            
            skin_lower = np.array([0, 20, 70], dtype=np.uint8)
            skin_upper = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
            is_occluded = cv2.countNonZero(skin_mask) / skin_mask.size
            skin_ratio = is_occluded
            is_occluded = 1.0 if is_occluded > 0.15 else 0.0
            
            return np.array([
                content_fullness,
                frame_quality,
                is_occluded,
                skin_ratio,
                float(edge_change),
                float(frame_diff_mean),
            ], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error extracting features: {e}")
            return np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    def extract_all_frames(self, video_path):
        """Extract features plus frame/edge diff signals from video"""
        logger.info(f"Extracting frames from: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None, None, None, None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        self.features = []
        self.frames_data = []
        diff_scores = []
        diff_mean_scores = []
        prev_gray = None
        prev_edges = None
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % 30 == 0:
                progress = frame_count / total_frames * 100
                print(f"\r  Processing: {frame_count}/{total_frames} frames ({progress:.1f}%)", end='', flush=True)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)

            diff_score = 0.0
            edge_change = 0.0
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                diff_score = float(np.mean(diff)) / 255.0
            if prev_edges is not None:
                edge_delta = cv2.absdiff(edges, prev_edges)
                edge_change = float(np.mean(edge_delta)) / 255.0
            prev_gray = gray
            prev_edges = edges

            features = self.extract_features(frame, edge_change=edge_change, frame_diff_mean=diff_score)
            self.features.append(features)
            diff_scores.append(diff_score)
            # simple short-window mean of diff for motion cue
            window_start = max(0, len(diff_scores) - 3)
            diff_mean = float(np.mean(diff_scores[window_start:])) if diff_scores else 0.0
            diff_mean_scores.append(diff_mean)
            
            self.frames_data.append({
                'frame_id': frame_count,
                'timestamp': frame_count / fps
            })
            
            frame_count += 1
        
        cap.release()
        
        self.features = np.array(self.features, dtype=np.float32)
        diff_scores = np.array(diff_scores, dtype=np.float32)
        diff_mean_scores = np.array(diff_mean_scores, dtype=np.float32)
        
        # Clear progress line and print summary
        print("\r" + " " * 80 + "\r", end='', flush=True)
        logger.info(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)")
        logger.info(f"  FPS: {fps:.1f}")
        logger.info(f"  Total frames: {frame_count}")
        
        return self.features, fps, duration, diff_scores
    
    def normalize_features(self, features, X_min=None, X_max=None):
        """Normalize features to [0, 1] range"""
        if X_min is None or X_max is None:
            X_min = features.min(axis=0)
            X_max = features.max(axis=0)
        
        # Avoid division by zero
        X_range = X_max - X_min
        X_range[X_range == 0] = 1.0
        
        normalized = (features - X_min) / X_range
        return normalized, X_min, X_max


def load_model_and_normalization(model_path='trained_model_gb_enriched_v2.pkl', 
                                  norm_path='model_gb_enriched_v2_normalization.pkl'):
    """Load trained model and normalization parameters"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        logger.info(f"Model loaded: {model_data['model'].__class__.__name__}")
        
        with open(norm_path, 'rb') as f:
            norm_data = pickle.load(f)
        logger.info(f"Normalization parameters loaded")
        
        return model_data['model'], norm_data['X_min'], norm_data['X_max']
    
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        return None, None, None


def smooth_probabilities(probs: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average smoothing over probabilities"""
    if window <= 1:
        return probs
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(probs, kernel, mode='same')


def detect_transitions(features, fps, model, X_min, X_max, confidence_threshold=0.15, smooth_window=5, diff_scores=None, diff_percentile=None, min_gap_seconds=1.5):
    """Detect transitions using trained model with optional temporal smoothing and diff gating"""
    logger.info("Detecting transitions...")
    
    # Normalize features
    normalized_features = (features - X_min) / (X_max - X_min + 1e-8)
    
    probabilities = model.predict_proba(normalized_features)
    raw_conf = probabilities[:, 1]
    smoothed_conf = smooth_probabilities(raw_conf, window=smooth_window)
    
    # Threshold on smoothed confidence
    positive_indices = np.where(smoothed_conf >= confidence_threshold)[0]
    logger.info(f"  Raw positives after smoothing: {len(positive_indices)}")

    # Optional frame-diff gating: keep only frames above percentile of diff_scores
    if diff_scores is not None and diff_percentile is not None:
        cutoff = np.percentile(diff_scores, diff_percentile)
        gated = [idx for idx in positive_indices if diff_scores[idx] >= cutoff]
        positive_indices = np.array(gated, dtype=int)
        logger.info(f"  After diff gating (pct>={diff_percentile}): {len(positive_indices)}")
    
    if len(positive_indices) == 0:
        logger.warning("  No transitions detected")
        return []
    
    # Cluster nearby detections (within 1 second = fps frames)
    if len(positive_indices) > 1:
        X_cluster = positive_indices.reshape(-1, 1)
        clustering = DBSCAN(eps=fps, min_samples=1).fit(X_cluster)
        labels = clustering.labels_
        
        transitions = []
        for cluster_id in np.unique(labels):
            cluster_indices = positive_indices[labels == cluster_id]
            # pick best by smoothed confidence
            best_idx = cluster_indices[np.argmax(smoothed_conf[cluster_indices])]
            best_confidence = smoothed_conf[best_idx]
            
            transitions.append({
                'frame_id': int(best_idx),
                'timestamp': best_idx / fps,
                'confidence': float(best_confidence)
            })
        transitions.sort(key=lambda x: x['timestamp'])
    else:
        transitions = [{
            'frame_id': int(positive_indices[0]),
            'timestamp': positive_indices[0] / fps,
            'confidence': float(smoothed_conf[positive_indices[0]])
        }]
    
    # Filter again by threshold (safety)
    transitions = [t for t in transitions if t['confidence'] >= confidence_threshold]

    # Apply min-gap suppression: keep top-confidence event per gap window
    if min_gap_seconds and min_gap_seconds > 0 and len(transitions) > 1:
        suppressed = []
        last_time = -1e9
        gap = min_gap_seconds
        # Sort by timestamp already
        for t in transitions:
            if t['timestamp'] - last_time >= gap:
                suppressed.append(t)
                last_time = t['timestamp']
            else:
                # within gap: keep the higher confidence between current and last kept
                if suppressed and t['confidence'] > suppressed[-1]['confidence']:
                    suppressed[-1] = t
                    last_time = t['timestamp']
        transitions = suppressed
    
    logger.info(f"  Detected transitions (after clustering): {len(transitions)}")
    return transitions


def format_timestamp(seconds):
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 100)
    return f"{minutes}:{secs:02d}.{milliseconds:02d}"


def save_results(video_path, transitions, fps, duration, output_dir=None):
    """Save detection results to JSON and TXT files"""
    if output_dir is None:
        output_dir = Path(video_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    video_name = Path(video_path).stem
    
    # Save as JSON
    json_path = output_dir / f"{video_name}_detected.json"
    json_results = {
        'video': video_name,
        'timestamp_generated': datetime.now().isoformat(),
        'video_duration_seconds': duration,
        'fps': fps,
        'total_transitions_detected': len(transitions),
        'transitions': transitions
    }
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Results saved to: {json_path}")
    
    # Save as TXT (timestamps only)
    txt_path = output_dir / f"{video_name}_detected.txt"
    with open(txt_path, 'w') as f:
        f.write(f"# Detected transitions for {video_name}\n")
        f.write(f"# Format: timestamp_in_seconds\n")
        f.write(f"# Total detected: {len(transitions)}\n")
        f.write(f"# Duration: {duration:.1f}s ({duration/60:.1f} min)\n\n")
        for t in transitions:
            f.write(f"{t['timestamp']:.2f}\n")
    
    logger.info(f"TXT results saved to: {txt_path}")
    
    return json_path, txt_path


def process_video(video_path, model=None, X_min=None, X_max=None, output_dir=None, confidence_threshold=0.15, smooth_window=5, diff_percentile=None, min_gap_seconds=1.5):
    """Main function to process a single video"""
    video_path = Path(video_path)
    
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return None
    
    logger.info(f"\n{'='*70}")
    logger.info(f"VIDEO: {video_path.name}")
    logger.info(f"{'='*70}\n")
    
    # Load model if not provided
    if model is None or X_min is None or X_max is None:
        model, X_min, X_max = load_model_and_normalization()
        if model is None:
            return None
    
    # Extract features
    extractor = FrameFeatureExtractor()
    features, fps, duration, diff_scores = extractor.extract_all_frames(str(video_path))
    
    if features is None:
        return None
    
    logger.info("")
    
    # Detect transitions
    transitions = detect_transitions(
        features,
        fps,
        model,
        X_min,
        X_max,
        confidence_threshold=confidence_threshold,
        smooth_window=smooth_window,
        diff_scores=diff_scores,
        diff_percentile=diff_percentile,
        min_gap_seconds=min_gap_seconds,
    )
    
    logger.info("")
    logger.info(f"RESULTS FOR {video_path.name}:")
    logger.info(f"  Total transitions: {len(transitions)}")
    if len(transitions) > 0:
        logger.info(f"  First 5 transitions:")
        for i, t in enumerate(transitions[:5], 1):
            logger.info(f"    {i}. {format_timestamp(t['timestamp'])} (confidence: {t['confidence']:.2f})")
        if len(transitions) > 5:
            logger.info(f"  ... and {len(transitions) - 5} more")
    
    logger.info("")
    
    # Save results
    json_path, txt_path = save_results(str(video_path), transitions, fps, duration, output_dir)
    
    return {
        'video': str(video_path),
        'transitions': transitions,
        'fps': fps,
        'duration': duration,
        'json_output': str(json_path),
        'txt_output': str(txt_path)
    }


def process_multiple_videos(video_dir, output_dir=None, confidence_threshold=0.15, smooth_window=5, model_path=None, norm_path=None, diff_percentile=None, min_gap_seconds=1.5):
    """Process all videos in a directory"""
    video_dir = Path(video_dir)
    if not video_dir.exists():
        logger.error(f"Directory not found: {video_dir}")
        return []
    
    # Load model once
    model, X_min, X_max = load_model_and_normalization(
        model_path or 'trained_model_gb_enriched_v2.pkl',
        norm_path or 'model_gb_enriched_v2_normalization.pkl'
    )
    if model is None:
        return []
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = [f for f in video_dir.glob('*') 
                   if f.suffix.lower() in video_extensions]
    
    if not video_files:
        logger.error(f"No video files found in: {video_dir}")
        return []
    
    logger.info(f"Found {len(video_files)} video(s) to process\n")
    
    results = []
    for i, video_path in enumerate(video_files, 1):
        logger.info(f"[{i}/{len(video_files)}] Processing...")
        result = process_video(
            str(video_path),
            model,
            X_min,
            X_max,
            output_dir,
            confidence_threshold=confidence_threshold,
            smooth_window=smooth_window,
            diff_percentile=diff_percentile,
            min_gap_seconds=min_gap_seconds,
        )
        if result:
            results.append(result)
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(f"SUMMARY - Processed {len(results)} video(s)")
    logger.info(f"{'='*70}")
    for result in results:
        logger.info(f"{Path(result['video']).name}: {len(result['transitions'])} transitions")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  Single video:  python detect_transitions_universal.py <video_path> [--out <dir>] [--thresh <0-1>] [--smooth <n>] [--model <pkl>] [--norm <pkl>]")
        print("  Multiple:      python detect_transitions_universal.py <video_dir> --dir [--out <dir>] [--thresh <0-1>] [--smooth <n>] [--model <pkl>] [--norm <pkl>]")
        print("\nExamples:")
        print("  python detect_transitions_universal.py data/testing_videos --dir --out results_enriched_v2/ --thresh 0.32 --smooth 5 --model trained_model_gb_enriched_v2.pkl --norm model_gb_enriched_v2_normalization.pkl")
        sys.exit(1)
    
    input_path = sys.argv[1]
    is_dir = False
    output_dir = None
    threshold = 0.15
    smooth_window = 5
    model_path = None
    norm_path = None
    diff_percentile = None
    min_gap_seconds = 1.5
    
    # Simple flag parser
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--dir':
            is_dir = True
            i += 1
        elif arg == '--out':
            if i + 1 < len(sys.argv):
                output_dir = sys.argv[i+1]
                i += 2
            else:
                print("Missing value for --out")
                sys.exit(1)
        elif arg in ('--thresh', '-t'):
            if i + 1 < len(sys.argv):
                threshold = float(sys.argv[i+1])
                i += 2
            else:
                print("Missing value for --thresh")
                sys.exit(1)
        elif arg == '--smooth':
            if i + 1 < len(sys.argv):
                smooth_window = int(sys.argv[i+1])
                i += 2
            else:
                print("Missing value for --smooth")
                sys.exit(1)
        elif arg == '--model':
            if i + 1 < len(sys.argv):
                model_path = sys.argv[i+1]
                i += 2
            else:
                print("Missing value for --model")
                sys.exit(1)
        elif arg == '--norm':
            if i + 1 < len(sys.argv):
                norm_path = sys.argv[i+1]
                i += 2
            else:
                print("Missing value for --norm")
                sys.exit(1)
        elif arg == '--diff-pct':
            if i + 1 < len(sys.argv):
                diff_percentile = float(sys.argv[i+1])
                i += 2
            else:
                print("Missing value for --diff-pct")
                sys.exit(1)
        elif arg == '--min-gap':
            if i + 1 < len(sys.argv):
                min_gap_seconds = float(sys.argv[i+1])
                i += 2
            else:
                print("Missing value for --min-gap")
                sys.exit(1)
        else:
            i += 1
    
    if is_dir:
        process_multiple_videos(
            input_path,
            output_dir,
            confidence_threshold=threshold,
            smooth_window=smooth_window,
            model_path=model_path,
            norm_path=norm_path,
            diff_percentile=diff_percentile,
            min_gap_seconds=min_gap_seconds,
        )
    else:
        # load model here if custom paths provided
        model, X_min, X_max = load_model_and_normalization(model_path or 'trained_model_gb_enriched_v2.pkl', norm_path or 'model_gb_enriched_v2_normalization.pkl')
        process_video(
            input_path,
            model=model,
            X_min=X_min,
            X_max=X_max,
            output_dir=output_dir,
            confidence_threshold=threshold,
            smooth_window=smooth_window,
            diff_percentile=diff_percentile,
            min_gap_seconds=min_gap_seconds,
        )
