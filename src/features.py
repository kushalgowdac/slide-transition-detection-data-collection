"""Handcrafted feature extraction utilities and caching helpers (stub).

Move feature-related code here when refactoring from root `main.py`.
"""
from pathlib import Path
import numpy as np
import cv2


class FeatureExtractor:
    """Wrapper around handcrafted features (keeps API similar to existing code)."""
    def compute_features(self, frame_paths):
        # Simple wrapper: re-use current main.py behaviour by reading images
        features = []
        prev = None
        for p in frame_paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            feat = {
                'frame_path': p,
                'mean_intensity': float(np.mean(img)),
                'std_intensity': float(np.std(img)),
            }
            if prev is not None:
                # placeholder for more features
                feat['pairwise'] = 0.0
            features.append(feat)
            prev = img
        return features
