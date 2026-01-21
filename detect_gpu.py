"""
GPU-Accelerated Slide Transition Detection
Supports both PyTorch deep learning model and sklearn models
"""
import argparse
import cv2
import numpy as np
from pathlib import Path
import pickle
import json
from tqdm import tqdm

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from detect_transitions_universal import FrameFeatureExtractor, detect_transitions


class SlideTransitionNet(nn.Module):
    """Same architecture as training script"""
    def __init__(self, input_dim=6, dropout=0.3):
        super(SlideTransitionNet, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x).squeeze()


def load_pytorch_model(model_path, device='cpu'):
    """Load PyTorch model"""
    checkpoint = torch.load(model_path, map_location=device)
    
    model = SlideTransitionNet(
        input_dim=checkpoint['input_dim'],
        dropout=checkpoint.get('dropout', 0.3)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['feature_cols']


def load_sklearn_model(model_path):
    """Load sklearn model"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def extract_features_gpu(video_path, device='cpu', batch_size=32):
    """
    Extract features from video with GPU acceleration
    Uses batched processing for better GPU utilization
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize feature extractor
    extractor = FrameFeatureExtractor()
    
    all_features = []
    frame_buffer = []
    
    print(f"Extracting features from {total_frames} frames (GPU batch size: {batch_size})...")
    
    with tqdm(total=total_frames) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_buffer.append(frame)
            
            # Process in batches
            if len(frame_buffer) >= batch_size:
                for f in frame_buffer:
                    features = extractor.extract_features(f)
                    all_features.append(features)
                frame_buffer = []
                pbar.update(batch_size)
        
        # Process remaining frames
        for f in frame_buffer:
            features = extractor.extract_features(f)
            all_features.append(features)
        pbar.update(len(frame_buffer))
    
    cap.release()
    
    # Convert to numpy array
    features_array = np.array(all_features)
    
    return features_array, fps


def predict_with_pytorch(model, features, scaler, device='cpu', batch_size=128):
    """
    Predict transitions using PyTorch model
    Uses batched inference for GPU efficiency
    """
    # Normalize features
    features_scaled = scaler.transform(features)
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(features_scaled)
    
    all_probs = []
    
    # Batch inference
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            probs = model(batch).cpu().numpy()
            all_probs.extend(probs)
    
    return np.array(all_probs)


def predict_with_sklearn(model, features, scaler):
    """Predict transitions using sklearn model"""
    features_scaled = scaler.transform(features)
    probs = model.predict_proba(features_scaled)[:, 1]
    return probs


def main():
    parser = argparse.ArgumentParser(description='GPU-accelerated slide transition detection')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--model', type=str, default='models/trained_model_deep.pth',
                        help='Path to model file (.pth for PyTorch, .pkl for sklearn)')
    parser.add_argument('--scaler', type=str, help='Path to scaler file (auto-detected if not specified)')
    parser.add_argument('--output', type=str, default='results_gpu',
                        help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.55,
                        help='Probability threshold for transition detection')
    parser.add_argument('--diff-percentile', type=float, default=90,
                        help='Frame difference percentile threshold')
    parser.add_argument('--min-gap', type=float, default=3.0,
                        help='Minimum gap between transitions (seconds)')
    parser.add_argument('--smooth-window', type=int, default=5,
                        help='Smoothing window size')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for inference')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU even if GPU is available')
    
    args = parser.parse_args()
    
    # Determine device
    if TORCH_AVAILABLE and not args.cpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")
    if device != 'cpu' and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Determine model type
    model_path = Path(args.model)
    is_pytorch = model_path.suffix == '.pth'
    
    # Auto-detect scaler path
    if args.scaler:
        scaler_path = args.scaler
    else:
        if is_pytorch:
            scaler_path = model_path.parent / 'model_deep_normalization.pkl'
        else:
            scaler_path = model_path.parent / f'{model_path.stem}_normalization.pkl'
    
    print(f"\nLoading model: {model_path}")
    print(f"Loading scaler: {scaler_path}")
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load model
    if is_pytorch:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch")
        model, feature_cols = load_pytorch_model(model_path, device)
        print(f"Loaded PyTorch model with {len(feature_cols)} features")
    else:
        model = load_sklearn_model(model_path)
        print(f"Loaded sklearn model: {type(model).__name__}")
    
    # Extract features
    print(f"\nProcessing video: {args.video}")
    features, fps = extract_features_gpu(args.video, device, batch_size=32)
    
    print(f"Extracted {len(features)} feature vectors")
    print(f"Video FPS: {fps:.2f}")
    
    # Predict probabilities
    print("\nPredicting transitions...")
    if is_pytorch:
        probs = predict_with_pytorch(model, features, scaler, device, args.batch_size)
    else:
        probs = predict_with_sklearn(model, features, scaler)
    
    # Detect transitions
    print("\nDetecting transitions...")
    timestamps = detect_transitions(
        probs=probs,
        fps=fps,
        threshold=args.threshold,
        diff_percentile=args.diff_percentile,
        min_gap_seconds=args.min_gap,
        smooth_window=args.smooth_window
    )
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_name = Path(args.video).stem
    
    # Save JSON
    result_json = {
        'video': str(args.video),
        'model': str(model_path),
        'model_type': 'pytorch' if is_pytorch else 'sklearn',
        'device': str(device),
        'parameters': {
            'threshold': args.threshold,
            'diff_percentile': args.diff_percentile,
            'min_gap': args.min_gap,
            'smooth_window': args.smooth_window
        },
        'fps': fps,
        'total_frames': len(features),
        'transitions': [float(t) for t in timestamps]
    }
    
    json_path = output_dir / f'{video_name}_detected.json'
    with open(json_path, 'w') as f:
        json.dump(result_json, f, indent=2)
    
    # Save TXT
    txt_path = output_dir / f'{video_name}_detected.txt'
    with open(txt_path, 'w') as f:
        for t in timestamps:
            f.write(f"{t:.2f}\n")
    
    print(f"\n{'='*60}")
    print(f"Detected {len(timestamps)} transitions")
    print(f"Results saved to: {output_dir}")
    print(f"  - {json_path.name}")
    print(f"  - {txt_path.name}")
    print(f"{'='*60}")
    
    # Show first few transitions
    if timestamps:
        print("\nFirst 5 transitions:")
        for t in timestamps[:5]:
            m, s = divmod(int(t), 60)
            print(f"  {m:02d}:{s:02d} ({t:.2f}s)")


if __name__ == '__main__':
    main()
