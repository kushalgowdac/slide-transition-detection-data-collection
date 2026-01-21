"""
Prepare toc_1 for training by extracting frames and creating labels
This converts your manual ground truth into training data
"""

import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm


def extract_frames_for_training(video_path, output_dir, fps=1.0):
    """Extract frames from video."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_skip = max(1, int(video_fps // fps))
    frames_data = []
    frame_idx = 0
    saved_count = 0
    
    print(f"Extracting frames from {video_path.name}...")
    print(f"  Video FPS: {video_fps:.2f}, Total frames: {total_frames:,}")
    
    pbar = tqdm(total=total_frames)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = frame_idx / video_fps
        
        if frame_idx % frame_skip == 0:
            frame_path = output_dir / f"frame_{saved_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frames_data.append({
                'frame_number': frame_idx,
                'timestamp': timestamp,
                'frame_path': str(frame_path),
                'video_name': video_path.stem
            })
            saved_count += 1
        
        frame_idx += 1
        pbar.update(1)
    
    cap.release()
    pbar.close()
    
    print(f"  Extracted {saved_count} frames\n")
    return frames_data


def compute_features(frame_path):
    """Compute features from frame."""
    try:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Content fullness
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        content_fullness = np.sum(binary > 0) / binary.size
        
        # 2. Frame quality
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = np.std(gray)
        frame_quality = (laplacian_var + contrast) / 2
        
        # 3. Occlusion
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 10, 60], dtype=np.uint8)
        upper_skin = np.array([20, 40, 100], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_pixels = np.sum(skin_mask > 0)
        skin_ratio = skin_pixels / skin_mask.size
        is_occluded = 1 if skin_ratio > 0.12 else 0
        
        return [content_fullness, frame_quality, is_occluded, skin_ratio]
    except:
        return None


def load_ground_truth(gt_file):
    """Load transitions and ideal frames."""
    transitions = []
    
    if not gt_file.exists():
        return []
    
    with open(gt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if '|' in line:
                    trans, ideal = line.split('|')
                    transitions.append(float(trans.strip()))
                else:
                    try:
                        transitions.append(float(line))
                    except:
                        pass
    
    return sorted(transitions)


def label_frames(frames_data, transitions, tolerance=1.0):
    """Label frames as transition (1) or not (0).
    
    A frame is labeled as transition if it's within `tolerance` seconds
    before the actual transition time.
    """
    labeled = []
    
    for frame_info in frames_data:
        timestamp = frame_info['timestamp']
        is_transition = 0
        
        # Check if this frame is near any transition
        for trans_time in transitions:
            # Mark frames in range [trans_time - tolerance, trans_time]
            if trans_time - tolerance <= timestamp <= trans_time:
                is_transition = 1
                break
        
        labeled.append({
            **frame_info,
            'is_transition_gt': is_transition
        })
    
    return labeled


def main():
    print("\n" + "=" * 80)
    print("   PREPARE TOC_1 FOR TRAINING")
    print("=" * 80 + "\n")
    
    # Paths
    video_path = Path('data/testing_videos/toc_1.mp4')
    gt_path = Path('data/testing_videos/toc_1_transitions.txt')
    output_dir = Path('data/frames/toc_1')
    
    # Validate
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}\n")
        return
    
    if not gt_path.exists():
        print(f"❌ Ground truth not found: {gt_path}\n")
        return
    
    # Step 1: Extract frames
    print("STEP 1: EXTRACT FRAMES")
    print("-" * 80)
    frames_data = extract_frames_for_training(video_path, output_dir, fps=1.0)
    
    # Step 2: Load ground truth
    print("STEP 2: LOAD GROUND TRUTH")
    print("-" * 80)
    transitions = load_ground_truth(gt_path)
    print(f"✅ Loaded {len(transitions)} transitions\n")
    
    # Step 3: Compute features
    print("STEP 3: COMPUTE FEATURES")
    print("-" * 80)
    print("Computing features for all frames...")
    
    for frame_info in tqdm(frames_data):
        features = compute_features(frame_info['frame_path'])
        if features:
            frame_info['content_fullness'] = features[0]
            frame_info['frame_quality'] = features[1]
            frame_info['is_occluded'] = features[2]
            frame_info['skin_ratio'] = features[3]
        else:
            frame_info['content_fullness'] = 0
            frame_info['frame_quality'] = 0
            frame_info['is_occluded'] = 0
            frame_info['skin_ratio'] = 0
    
    print()
    
    # Step 4: Label frames
    print("STEP 4: LABEL FRAMES")
    print("-" * 80)
    labeled_frames = label_frames(frames_data, transitions, tolerance=1.0)
    
    transition_count = sum(1 for f in labeled_frames if f['is_transition_gt'] == 1)
    print(f"✅ Labeled {len(labeled_frames)} frames")
    print(f"   - Transitions: {transition_count}")
    print(f"   - Non-transitions: {len(labeled_frames) - transition_count}\n")
    
    # Step 5: Save as CSV
    print("STEP 5: SAVE TRAINING DATA")
    print("-" * 80)
    
    df = pd.DataFrame(labeled_frames)
    df = df[[
        'frame_number', 'timestamp', 'frame_path', 'video_name',
        'content_fullness', 'frame_quality', 'is_occluded', 'skin_ratio',
        'is_transition_gt'
    ]]
    
    csv_path = Path('toc_1_training_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved to {csv_path}\n")
    
    # Summary
    print("NEXT STEPS:")
    print("-" * 80)
    print("1. Review the CSV file to verify labels")
    print("2. Merge with existing training data:")
    print("   - Append toc_1_training_data.csv to labeled_dataset.csv")
    print("3. Retrain model with combined dataset:")
    print("   - python train_classifier.py")
    print("4. Test again:")
    print("   - python test_model_professional.py")
    print()


if __name__ == '__main__':
    main()
