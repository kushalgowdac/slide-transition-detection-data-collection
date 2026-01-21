"""
Compare Detected Transitions with Ground Truth
Calculates precision, recall, F1 score, and other metrics
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TransitionComparator:
    """Compare detected transitions with ground truth"""
    
    def __init__(self, tolerance_seconds=0.5):
        """
        Args:
            tolerance_seconds: How close detected must be to ground truth (default 0.5s)
        """
        self.tolerance = tolerance_seconds
        self.ground_truth = []
        self.detected = []
        self.tp = 0  # True positives
        self.fp = 0  # False positives
        self.fn = 0  # False negatives
    
    def load_ground_truth(self, file_path) -> bool:
        """Load ground truth transitions from file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"Ground truth file not found: {file_path}")
            return False
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            self.ground_truth = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Handle both formats:
                # 1. Simple: "1.41"
                # 2. Pipe: "1.41|1.40"
                if '|' in line:
                    timestamp = float(line.split('|')[0].strip())
                else:
                    timestamp = float(line)
                
                self.ground_truth.append(timestamp)
            
            self.ground_truth.sort()
            logger.info(f"Loaded {len(self.ground_truth)} ground truth transitions from {file_path.name}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading ground truth: {e}")
            return False
    
    def load_detected(self, file_path) -> bool:
        """Load detected transitions from JSON or TXT file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"Detected file not found: {file_path}")
            return False
        
        try:
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                self.detected = [t['timestamp'] for t in data.get('transitions', [])]
            else:
                # TXT format
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                self.detected = []
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    try:
                        self.detected.append(float(line))
                    except ValueError:
                        continue
            
            self.detected.sort()
            logger.info(f"Loaded {len(self.detected)} detected transitions from {file_path.name}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading detected transitions: {e}")
            return False
    
    def find_matches(self) -> Dict:
        """Find matches between detected and ground truth within tolerance"""
        matched_gt = set()
        matched_det = set()
        
        for i, det in enumerate(self.detected):
            for j, gt in enumerate(self.ground_truth):
                if j in matched_gt:
                    continue
                
                if abs(det - gt) <= self.tolerance:
                    matched_gt.add(j)
                    matched_det.add(i)
                    break
        
        self.tp = len(matched_gt)
        self.fp = len(self.detected) - self.tp
        self.fn = len(self.ground_truth) - self.tp
        
        return {
            'matched_ground_truth': sorted(matched_gt),
            'matched_detected': sorted(matched_det)
        }
    
    def calculate_metrics(self) -> Dict:
        """Calculate precision, recall, F1 score"""
        metrics = {}
        
        # Precision: TP / (TP + FP)
        if self.tp + self.fp > 0:
            metrics['precision'] = self.tp / (self.tp + self.fp)
        else:
            metrics['precision'] = 0.0
        
        # Recall: TP / (TP + FN)
        if self.tp + self.fn > 0:
            metrics['recall'] = self.tp / (self.tp + self.fn)
        else:
            metrics['recall'] = 0.0
        
        # F1 Score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / \
                          (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0.0
        
        # Accuracy (of detection rate)
        metrics['detection_rate'] = self.tp / len(self.ground_truth) if self.ground_truth else 0.0
        metrics['false_positive_rate'] = self.fp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        
        return metrics
    
    def get_unmatched(self) -> Tuple[List, List]:
        """Get unmatched ground truth and detected transitions"""
        matches = self.find_matches()
        
        unmatched_gt = [self.ground_truth[i] for i in range(len(self.ground_truth))
                        if i not in matches['matched_ground_truth']]
        unmatched_det = [self.detected[i] for i in range(len(self.detected))
                         if i not in matches['matched_detected']]
        
        return unmatched_gt, unmatched_det
    
    def compare(self, ground_truth_file: str, detected_file: str) -> Dict:
        """Full comparison pipeline"""
        logger.info(f"\n{'='*70}")
        logger.info(f"COMPARING DETECTED vs GROUND TRUTH")
        logger.info(f"{'='*70}\n")
        
        if not self.load_ground_truth(ground_truth_file):
            return None
        
        if not self.load_detected(detected_file):
            return None
        
        logger.info(f"\nTolerance: ±{self.tolerance} seconds\n")
        
        # Find matches
        matches = self.find_matches()
        logger.info(f"Matches found: {self.tp} (within ±{self.tolerance}s)")
        logger.info(f"False positives: {self.fp}")
        logger.info(f"False negatives: {self.fn}\n")
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        logger.info(f"METRICS:")
        logger.info(f"  Precision (correct detections):      {metrics['precision']:.2%}")
        logger.info(f"  Recall (found ground truth):         {metrics['recall']:.2%}")
        logger.info(f"  F1 Score (balanced metric):          {metrics['f1']:.2%}")
        logger.info(f"  Detection Rate:                      {metrics['detection_rate']:.2%}")
        logger.info(f"  False Positive Rate:                 {metrics['false_positive_rate']:.2%}\n")
        
        # Unmatched transitions
        unmatched_gt, unmatched_det = self.get_unmatched()
        
        if unmatched_gt:
            logger.info(f"MISSED TRANSITIONS ({len(unmatched_gt)}):")
            for ts in unmatched_gt[:10]:
                logger.info(f"  {self.format_time(ts)} - No detection nearby")
            if len(unmatched_gt) > 10:
                logger.info(f"  ... and {len(unmatched_gt) - 10} more\n")
        
        if unmatched_det:
            logger.info(f"FALSE POSITIVES ({len(unmatched_det)}):")
            for ts in unmatched_det[:10]:
                logger.info(f"  {self.format_time(ts)} - No ground truth nearby")
            if len(unmatched_det) > 10:
                logger.info(f"  ... and {len(unmatched_det) - 10} more\n")
        
        # Summary
        logger.info(f"{'='*70}")
        logger.info(f"SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Ground Truth:    {len(self.ground_truth)} transitions")
        logger.info(f"Detected:        {len(self.detected)} transitions")
        logger.info(f"Matched:         {self.tp} transitions")
        logger.info(f"Accuracy:        {metrics['precision']:.1%} precision, {metrics['recall']:.1%} recall")
        
        return {
            'ground_truth_count': len(self.ground_truth),
            'detected_count': len(self.detected),
            'true_positives': self.tp,
            'false_positives': self.fp,
            'false_negatives': self.fn,
            'metrics': metrics,
            'unmatched_ground_truth': unmatched_gt,
            'unmatched_detected': unmatched_det,
            'tolerance': self.tolerance
        }
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Format seconds to MM:SS.MS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 100)
        return f"{minutes}:{secs:02d}.{ms:02d}"


def find_ground_truth_file(video_name: str, search_dirs=None) -> str:
    """Find ground truth file for a video"""
    if search_dirs is None:
        search_dirs = [
            'data/ground_truth',
            'data/testing_videos',
            Path.cwd()
        ]
    
    # Try different naming patterns
    patterns = [
        f"{video_name}_transitions.txt",
        f"{video_name}.txt",
        "transitions.txt"
    ]
    
    for search_dir in search_dirs:
        search_dir = Path(search_dir)
        if not search_dir.exists():
            continue
        
        for pattern in patterns:
            possible_file = search_dir / pattern
            if possible_file.exists():
                return str(possible_file)
        
        # Check subdirectories too
        for subdir in search_dir.iterdir():
            if subdir.is_dir():
                for pattern in patterns:
                    possible_file = subdir / pattern
                    if possible_file.exists():
                        return str(possible_file)
    
    return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("\nUsage:")
        print("  python compare_with_ground_truth.py <detected_file> <ground_truth_file> [tolerance]")
        print("\nExamples:")
        print("  python compare_with_ground_truth.py algo_1_detected.json data/testing_videos/algo_1_transitions.txt")
        print("  python compare_with_ground_truth.py results/algo_1_detected.json data/testing_videos/algo_1_transitions.txt 0.5")
        print("\nOr let it auto-find ground truth:")
        print("  python compare_with_ground_truth.py algo_1_detected.json")
        sys.exit(1)
    
    detected_file = sys.argv[1]
    ground_truth_file = sys.argv[2] if len(sys.argv) > 2 else None
    tolerance = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    
    # Auto-find ground truth if not provided
    if not ground_truth_file:
        video_name = Path(detected_file).stem.replace('_detected', '')
        ground_truth_file = find_ground_truth_file(video_name)
        
        if ground_truth_file:
            logger.info(f"Auto-found ground truth: {ground_truth_file}\n")
        else:
            logger.error(f"Could not find ground truth file for: {video_name}")
            sys.exit(1)
    
    # Run comparison
    comparator = TransitionComparator(tolerance_seconds=tolerance)
    results = comparator.compare(ground_truth_file, detected_file)
    
    if results:
        # Optionally save results to JSON
        output_json = Path(detected_file).stem + "_comparison.json"
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nComparison results saved to: {output_json}")
