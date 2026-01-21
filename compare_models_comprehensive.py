#!/usr/bin/env python3
"""
Comprehensive Model Comparison: Original vs Improved sklearn Model
Tests on multiple new teacher videos to measure generalization improvement.
"""

import sys
sys.path.insert(0, 'src')

import pickle
import numpy as np
import cv2
from pathlib import Path
import logging
from datetime import datetime
import json

# Import SimpleDecisionTree to enable unpickling
try:
    from train_classifier import SimpleDecisionTree
except:
    try:
        from train_classifier_v2 import SimpleDecisionTree
    except:
        # Define a dummy class for unpickling
        class SimpleDecisionTree:
            pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

FEATURE_NAMES = ['content_fullness', 'frame_quality', 'is_occluded', 'skin_ratio']

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

class FrameFeatureExtractor:
    """Extract features from frames."""
    
    @staticmethod
    def extract_features(frame):
        """Extract features from a frame."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # Content fullness (variance of grayscale)
            content_fullness = np.var(gray) / 10000.0
            content_fullness = min(1.0, max(0.0, content_fullness))
            
            # Frame quality (sharpness using Laplacian)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            frame_quality = np.var(laplacian) / 1000.0
            frame_quality = min(1.0, max(0.0, frame_quality))
            
            # Skin detection (is_occluded proxy)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 10, 60], dtype=np.uint8)
            upper_skin = np.array([20, 150, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_ratio = np.sum(skin_mask > 0) / (height * width)
            
            # Is occluded (if high skin ratio, likely occluded)
            is_occluded = 1.0 if skin_ratio > 0.15 else 0.0
            
            return {
                'content_fullness': content_fullness,
                'frame_quality': frame_quality,
                'is_occluded': is_occluded,
                'skin_ratio': skin_ratio
            }
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None


# ============================================================================
# MODEL TESTING
# ============================================================================

def load_model(model_path):
    """Load model and normalization."""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict) and 'model' in data:
        model = data['model']
        X_min = data.get('X_min')
        X_max = data.get('X_max')
    else:
        model = data
        X_min = None
        X_max = None
    
    return model, X_min, X_max


def extract_frames_and_features(video_path, fps=1.0):
    """Extract frames and features from video."""
    logger.info(f"Processing: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / video_fps if video_fps > 0 else 0
    
    logger.info(f"  Video: {video_duration:.1f}s, {video_fps:.1f} fps, {total_frames:,} frames")
    
    frame_skip = max(1, int(video_fps / fps))
    logger.info(f"  Sampling: every {frame_skip} frames (target {fps} fps)")
    
    frame_data = []
    frame_idx = 0
    extracted = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_skip == 0:
            timestamp = frame_idx / video_fps
            features = FrameFeatureExtractor.extract_features(frame)
            if features:
                features['timestamp'] = timestamp
                features['frame_idx'] = frame_idx
                frame_data.append(features)
                extracted += 1
        
        frame_idx += 1
    
    cap.release()
    
    logger.info(f"  Extracted: {extracted} frames")
    return frame_data


def normalize_features(features, X_min, X_max):
    """Normalize feature vector."""
    feature_array = np.array([
        features['content_fullness'],
        features['frame_quality'],
        features['is_occluded'],
        features['skin_ratio']
    ])
    
    if X_min is not None and X_max is not None:
        feature_array = (feature_array - X_min) / (X_max - X_min + 1e-8)
    
    return feature_array.reshape(1, -1)


def detect_transitions(model, frame_data, X_min=None, X_max=None):
    """Detect transitions using model."""
    predictions = []
    
    for frame_features in frame_data:
        X = normalize_features(frame_features, X_min, X_max)
        
        if hasattr(model, 'predict_proba'):
            # sklearn model
            prob = model.predict_proba(X)[0][1]
        else:
            # custom model
            prob = model.predict(X)[0]
        
        predictions.append({
            'timestamp': frame_features['timestamp'],
            'probability': float(prob),
            'prediction': 1 if prob > 0.5 else 0
        })
    
    # Simple clustering: group consecutive positive predictions
    transitions = []
    in_transition = False
    transition_start = None
    transition_probs = []
    
    for pred in predictions:
        if pred['prediction'] == 1:
            if not in_transition:
                in_transition = True
                transition_start = pred['timestamp']
                transition_probs = [pred['probability']]
            else:
                transition_probs.append(pred['probability'])
        else:
            if in_transition:
                # Transition ended
                transition_time = transition_start + (len(transition_probs) - 1) * 0.5
                avg_prob = np.mean(transition_probs)
                transitions.append({
                    'timestamp': transition_time,
                    'probability': avg_prob,
                    'duration': len(transition_probs)
                })
                in_transition = False
                transition_probs = []
    
    # Handle last transition
    if in_transition:
        transition_time = transition_start + (len(transition_probs) - 1) * 0.5
        avg_prob = np.mean(transition_probs)
        transitions.append({
            'timestamp': transition_time,
            'probability': avg_prob,
            'duration': len(transition_probs)
        })
    
    return transitions


def test_model(model_name, model_path, video_path):
    """Test a single model on a video."""
    logger.info(f"\n" + "="*70)
    logger.info(f"Testing: {model_name}")
    logger.info("="*70)
    
    try:
        # Load model
        model, X_min, X_max = load_model(model_path)
        logger.info(f"Model loaded: {type(model).__name__}")
        
        # Extract frames
        frame_data = extract_frames_and_features(video_path, fps=1.0)
        
        # Detect transitions
        logger.info("\nDetecting transitions...")
        transitions = detect_transitions(model, frame_data, X_min, X_max)
        
        logger.info(f"Detected: {len(transitions)} transitions")
        for i, t in enumerate(transitions[:5], 1):
            logger.info(f"  {i}. {t['timestamp']:.1f}s (prob={t['probability']:.2f})")
        if len(transitions) > 5:
            logger.info(f"  ... and {len(transitions) - 5} more")
        
        return transitions
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return []


# ============================================================================
# MAIN COMPARISON
# ============================================================================

def main():
    """Run comprehensive model comparison."""
    logger.info("\n" + "="*80)
    logger.info("MODEL IMPROVEMENT COMPARISON TEST")
    logger.info("="*80)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    # Define test cases
    test_cases = [
        ("algo_1.mp4", "New teacher (Algorithms)"),
        ("cn_1.mp4", "New teacher (Competitive)"),
        ("toc_1.mp4", "New teacher (TOC)"),
    ]
    
    results = {}
    
    for video_name, description in test_cases:
        video_path = Path(f"data/testing_videos/{video_name}")
        
        if not video_path.exists():
            logger.warning(f"Skipping {video_name} - file not found")
            continue
        
        logger.info(f"\n" + "="*80)
        logger.info(f"VIDEO: {video_name} ({description})")
        logger.info("="*80)
        
        # Test original model
        original_transitions = test_model(
            "ORIGINAL (Custom Decision Tree)",
            "trained_model.pkl",
            str(video_path)
        )
        
        # Test improved model
        improved_transitions = test_model(
            "IMPROVED (sklearn with class balancing)",
            "trained_model_sklearn_v3.pkl",
            str(video_path)
        )
        
        results[video_name] = {
            'description': description,
            'original_detections': len(original_transitions),
            'improved_detections': len(improved_transitions),
            'improvement': len(improved_transitions) - len(original_transitions)
        }
        
        logger.info(f"\n" + "-"*70)
        logger.info(f"COMPARISON FOR {video_name}:")
        logger.info("-"*70)
        logger.info(f"  Original model: {len(original_transitions):3d} transitions detected")
        logger.info(f"  Improved model: {len(improved_transitions):3d} transitions detected")
        logger.info(f"  Difference:    {len(improved_transitions) - len(original_transitions):+3d}")
    
    # Summary
    logger.info(f"\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    for video_name, data in results.items():
        logger.info(f"\n{video_name} ({data['description']}):")
        logger.info(f"  Original: {data['original_detections']} detections")
        logger.info(f"  Improved: {data['improved_detections']} detections")
        logger.info(f"  Change:   {data['improvement']:+d}")
    
    # Save results
    results['timestamp'] = datetime.now().isoformat()
    with open('model_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to model_comparison_results.json")
    
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS")
    logger.info("="*80)
    logger.info("""
The improved model (sklearn with class balancing) should:
1. Detect MORE transitions on new teachers (better recall)
2. Have more consistent detection across different videos
3. Not suffer from overfitting like the original model

Key metrics from earlier testing:
- Original model: 97.45% accuracy on test set, but 0% recall on new videos (overfitting!)
- Improved model: 80.25% recall on test set, should generalize much better

Next steps:
1. Verify detections are correct by sampling frames
2. If improved model detects too many false positives, adjust confidence threshold
3. Consider collecting more training data from these new teachers
""")


if __name__ == '__main__':
    main()
