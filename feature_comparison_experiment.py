"""
Feature Comparison Experiment: Test different feature combinations
This script tests how different features affect model performance.

Current Features (4):
  1. content_fullness (45% weight)
  2. frame_quality (33% weight)
  3. is_occluded (15% weight)
  4. skin_ratio (7% weight)

Proposed Features to Test:
  5. ssim_score (structural similarity)
  6. histogram_distance (color distribution)
  7. edge_density (proportion of edge pixels)

Usage:
  python feature_comparison_experiment.py --dataset labeled_dataset.csv --model trained_model.pkl
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pickle
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SCIKIT_IMAGE = True
except ImportError:
    HAS_SCIKIT_IMAGE = False
    print("⚠️  scikit-image not installed. Install with: pip install scikit-image")
    print("   SSIM feature will be skipped.")


class FeatureComparison:
    """Compare different feature combinations"""
    
    def __init__(self, dataset_path='labeled_dataset.csv'):
        """Initialize with dataset"""
        self.dataset_path = Path(dataset_path)
        self.df = None
        self.results = {}
        
    def load_dataset(self):
        """Load and validate dataset"""
        print(f"Loading dataset from {self.dataset_path}...")
        self.df = pd.read_csv(self.dataset_path)
        print(f"✓ Loaded {len(self.df)} frames")
        
        # Check required columns
        required = ['content_fullness', 'frame_quality', 'is_occluded', 'skin_ratio', 'is_transition_gt']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        # Check frame paths exist
        if 'frame_path' in self.df.columns:
            existing = self.df['frame_path'].apply(lambda x: Path(x).exists() if isinstance(x, str) else False).sum()
            print(f"✓ Frame paths: {existing}/{len(self.df)} exist")
    
    def add_ssim_feature(self):
        """Add SSIM (Structural Similarity) feature"""
        if not HAS_SCIKIT_IMAGE:
            print("⚠️  Skipping SSIM feature (scikit-image not installed)")
            return
        
        print("\n📊 Computing SSIM scores...")
        ssim_scores = []
        
        for idx in tqdm(range(len(self.df)), desc="SSIM"):
            if idx == 0:
                ssim_scores.append(0.0)  # First frame has no previous
                continue
            
            curr_path = self.df.iloc[idx]['frame_path']
            prev_path = self.df.iloc[idx-1]['frame_path']
            
            # Only compute SSIM if frames are consecutive
            if self.df.iloc[idx]['frame_idx'] - self.df.iloc[idx-1]['frame_idx'] != 1:
                ssim_scores.append(0.0)
                continue
            
            try:
                curr_img = cv2.imread(curr_path, cv2.IMREAD_GRAYSCALE)
                prev_img = cv2.imread(prev_path, cv2.IMREAD_GRAYSCALE)
                
                if curr_img is None or prev_img is None:
                    ssim_scores.append(0.0)
                    continue
                
                # Ensure same size
                if curr_img.shape != prev_img.shape:
                    h, w = min(curr_img.shape[0], prev_img.shape[0]), \
                            min(curr_img.shape[1], prev_img.shape[1])
                    curr_img = cv2.resize(curr_img, (w, h))
                    prev_img = cv2.resize(prev_img, (w, h))
                
                score = ssim(prev_img, curr_img, data_range=255)
                # Convert from [-1, 1] to [0, 1]
                normalized_score = (score + 1) / 2
                ssim_scores.append(normalized_score)
            except Exception as e:
                ssim_scores.append(0.0)
        
        self.df['ssim_score'] = ssim_scores
        print(f"✓ SSIM computed. Mean: {np.mean(ssim_scores):.4f}")
    
    def add_histogram_feature(self):
        """Add histogram distance feature"""
        print("\n📊 Computing histogram distances...")
        hist_dists = []
        
        for idx in tqdm(range(len(self.df)), desc="Histogram"):
            if idx == 0:
                hist_dists.append(0.0)
                continue
            
            curr_path = self.df.iloc[idx]['frame_path']
            prev_path = self.df.iloc[idx-1]['frame_path']
            
            # Only compute if consecutive frames
            if self.df.iloc[idx]['frame_idx'] - self.df.iloc[idx-1]['frame_idx'] != 1:
                hist_dists.append(0.0)
                continue
            
            try:
                curr_img = cv2.imread(curr_path, cv2.IMREAD_GRAYSCALE)
                prev_img = cv2.imread(prev_path, cv2.IMREAD_GRAYSCALE)
                
                if curr_img is None or prev_img is None:
                    hist_dists.append(0.0)
                    continue
                
                hist1 = cv2.calcHist([prev_img], [0], None, [256], [0, 256])
                hist2 = cv2.calcHist([curr_img], [0], None, [256], [0, 256])
                
                hist1 = cv2.normalize(hist1, hist1).flatten()
                hist2 = cv2.normalize(hist2, hist2).flatten()
                
                distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
                hist_dists.append(distance)
            except Exception:
                hist_dists.append(0.0)
        
        self.df['histogram_distance'] = hist_dists
        print(f"✓ Histogram distances computed. Mean: {np.mean(hist_dists):.4f}")
    
    def add_edge_density_feature(self):
        """Add edge density feature"""
        print("\n📊 Computing edge densities...")
        edge_densities = []
        
        for idx in tqdm(range(len(self.df)), desc="Edge Density"):
            frame_path = self.df.iloc[idx]['frame_path']
            
            try:
                img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    edge_densities.append(0.0)
                    continue
                
                # Canny edge detection
                edges = cv2.Canny(img, 100, 200)
                edge_density = float(np.count_nonzero(edges) / edges.size)
                edge_densities.append(edge_density)
            except Exception:
                edge_densities.append(0.0)
        
        self.df['edge_density'] = edge_densities
        print(f"✓ Edge densities computed. Mean: {np.mean(edge_densities):.4f}")
    
    def train_and_evaluate(self, features, name):
        """Train model with given features and evaluate"""
        print(f"\n🤖 Testing feature set: {name}")
        print(f"   Features: {features}")
        
        # Check if all features exist
        missing = [f for f in features if f not in self.df.columns]
        if missing:
            print(f"   ⚠️  Skipping - missing features: {missing}")
            return None
        
        X = self.df[features].values
        y = self.df['is_transition_gt'].values
        
        # Train-test split (70-30)
        split_idx = int(0.7 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train Decision Tree
        clf = DecisionTreeClassifier(max_depth=15, random_state=42)
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary', zero_division=0
        )
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        result = {
            'features': features,
            'name': name,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'positive_samples': np.sum(y_test),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        }
        
        self.results[name] = result
        
        # Print results
        print(f"   Accuracy: {result['accuracy']:.4f}")
        print(f"   Precision: {result['precision']:.4f}")
        print(f"   Recall: {result['recall']:.4f}")
        print(f"   F1: {result['f1']:.4f}")
        print(f"   Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
        
        return result
    
    def compare_all_features(self):
        """Run all feature combination experiments"""
        print("\n" + "="*70)
        print("FEATURE COMPARISON EXPERIMENT")
        print("="*70)
        
        # Baseline: Current 4 features
        self.train_and_evaluate(
            ['content_fullness', 'frame_quality', 'is_occluded', 'skin_ratio'],
            'Baseline (Current 4 Features)'
        )
        
        # Try adding SSIM
        if 'ssim_score' in self.df.columns:
            self.train_and_evaluate(
                ['content_fullness', 'frame_quality', 'is_occluded', 'skin_ratio', 'ssim_score'],
                'Baseline + SSIM'
            )
        
        # Try adding Histogram
        if 'histogram_distance' in self.df.columns:
            self.train_and_evaluate(
                ['content_fullness', 'frame_quality', 'is_occluded', 'skin_ratio', 'histogram_distance'],
                'Baseline + Histogram'
            )
        
        # Try adding Edge Density
        if 'edge_density' in self.df.columns:
            self.train_and_evaluate(
                ['content_fullness', 'frame_quality', 'is_occluded', 'skin_ratio', 'edge_density'],
                'Baseline + Edge Density'
            )
        
        # Try simplified (just top 2 features)
        self.train_and_evaluate(
            ['content_fullness', 'frame_quality'],
            'Simplified (Top 2 Features)'
        )
        
        # Try all features combined
        all_features = ['content_fullness', 'frame_quality', 'is_occluded', 'skin_ratio']
        if 'ssim_score' in self.df.columns:
            all_features.append('ssim_score')
        if 'histogram_distance' in self.df.columns:
            all_features.append('histogram_distance')
        if 'edge_density' in self.df.columns:
            all_features.append('edge_density')
        
        if len(all_features) > 4:
            self.train_and_evaluate(all_features, 'All Features Combined')
    
    def print_comparison_table(self):
        """Print comparison table"""
        print("\n" + "="*70)
        print("COMPARISON RESULTS")
        print("="*70)
        print(f"{'Configuration':<35} {'Accuracy':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 70)
        
        for name, result in self.results.items():
            print(f"{name:<35} {result['accuracy']:.4f}         {result['recall']:.4f}         {result['f1']:.4f}")
        
        print("\n📊 KEY FINDINGS:")
        baseline_f1 = self.results.get('Baseline (Current 4 Features)', {}).get('f1', 0)
        for name, result in sorted(self.results.items(), key=lambda x: x[1]['f1'], reverse=True):
            if name != 'Baseline (Current 4 Features)':
                improvement = (result['f1'] - baseline_f1) * 100
                sign = "↑" if improvement > 0 else "↓"
                print(f"  {sign} {name}: {improvement:+.2f}% F1 change")


def main():
    parser = argparse.ArgumentParser(description='Compare feature combinations')
    parser.add_argument('--dataset', default='labeled_dataset.csv', help='Dataset CSV file')
    parser.add_argument('--no-ssim', action='store_true', help='Skip SSIM computation')
    parser.add_argument('--no-histogram', action='store_true', help='Skip histogram computation')
    parser.add_argument('--no-edges', action='store_true', help='Skip edge density computation')
    args = parser.parse_args()
    
    # Initialize
    comp = FeatureComparison(args.dataset)
    comp.load_dataset()
    
    # Add features
    if not args.no_ssim:
        comp.add_ssim_feature()
    if not args.no_histogram:
        comp.add_histogram_feature()
    if not args.no_edges:
        comp.add_edge_density_feature()
    
    # Compare
    comp.compare_all_features()
    comp.print_comparison_table()
    
    print("\n✅ Feature comparison complete!")
    print("\n💡 INTERPRETATION:")
    print("  • Baseline: Your current model with 4 features")
    print("  • Higher Recall = Better at detecting transitions")
    print("  • Higher F1 = Better overall performance")
    print("  • If all similar: Features are well-chosen, won't improve much")


if __name__ == '__main__':
    main()
