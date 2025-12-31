"""
Lecture Video Frame Extraction and Feature Computation
Author: Person A
Week 1 Deliverable
"""

import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import json
import argparse
import sys
import hashlib
import datetime
import random

class LectureFrameExtractor:
    def __init__(self, output_dir='data/frames', resize=None, color_mode='color'):
        """output_dir: base path where per-video folders are created
        resize: tuple (w,h) or None
        color_mode: 'color' or 'gray' - determines saved image mode
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resize = resize
        self.color_mode = color_mode
        
    def extract_frames(self, video_path, fps=1.0, dense_threshold=0.3):
        """
        Extract frames from video at specified FPS + dense sampling near changes
        
        Args:
            video_path: Path to video file
            fps: Base frames per second to extract
            dense_threshold: Histogram difference threshold for dense sampling
        
        Returns:
            DataFrame with frame metadata
        """
        video_path = Path(video_path)
        video_name = video_path.stem
        
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Validate FPS and calculate frame skip (ensure >= 1)
        if not video_fps or video_fps <= 0:
            # fallback to requested fps or 1.0 if that's also invalid
            video_fps = max(1.0, float(fps))

        frame_skip = max(1, int(video_fps // fps))
        
        frames_data = []
        prev_gray = None
        frame_idx = 0
        saved_count = 0
        
        print(f"Processing {video_name} ({total_frames} frames @ {video_fps} fps)...")
        
        pbar = tqdm(total=total_frames)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / video_fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            should_save = (frame_idx % frame_skip == 0)
            
            # Dense sampling near changes
            is_transition = False
            if prev_gray is not None:
                hist_diff = self._histogram_diff(prev_gray, gray)
                if hist_diff > dense_threshold:
                    should_save = True
                    is_transition = True
            
            if should_save:
                # Prepare frame (resize / color_mode) then save
                frame_out = frame
                if self.resize:
                    try:
                        frame_out = cv2.resize(frame_out, (self.resize[0], self.resize[1]), interpolation=cv2.INTER_AREA)
                    except Exception:
                        pass

                if self.color_mode == 'gray':
                    save_img = cv2.cvtColor(frame_out, cv2.COLOR_BGR2GRAY)
                else:
                    save_img = frame_out

                # Save frame
                frame_filename = f"{video_name}_frame_{saved_count:05d}_{timestamp:.2f}s.jpg"
                frame_path = self.output_dir / video_name / frame_filename
                frame_path.parent.mkdir(parents=True, exist_ok=True)

                # encode image to memory to compute MD5 and write bytes (avoids a second read)
                ext = '.jpg'
                try:
                    success, buf = cv2.imencode(ext, save_img)
                    if success:
                        img_bytes = buf.tobytes()
                        md5sum = hashlib.md5(img_bytes).hexdigest()
                        with open(frame_path, 'wb') as fh:
                            fh.write(img_bytes)
                    else:
                        # fallback
                        cv2.imwrite(str(frame_path), save_img)
                        try:
                            with open(frame_path, 'rb') as fh:
                                md5sum = hashlib.md5(fh.read()).hexdigest()
                        except Exception:
                            md5sum = ''
                except Exception:
                    # robust fallback
                    cv2.imwrite(str(frame_path), save_img)
                    try:
                        with open(frame_path, 'rb') as fh:
                            md5sum = hashlib.md5(fh.read()).hexdigest()
                    except Exception:
                        md5sum = ''

                frames_data.append({
                    'video_name': video_name,
                    'frame_path': str(frame_path),
                    'frame_idx': frame_idx,
                    'timestamp': timestamp,
                    'saved_count': saved_count,
                    'is_transition': bool(is_transition),
                    'video_fps': float(video_fps),
                    'video_total_frames': int(total_frames),
                    'source_video': str(video_path)
                })
                saved_count += 1
            
            prev_gray = gray
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        df = pd.DataFrame(frames_data)
        print(f"Extracted {saved_count} frames from {video_name}")
        
        return df
    
    def _histogram_diff(self, img1, img2):
        """Calculate Bhattacharyya distance between histograms"""
        hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
        
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)


class FeatureExtractor:
    """Extract handcrafted features from frame pairs"""
    
    def compute_features(self, frame_paths):
        """
        Compute features for consecutive frame pairs
        
        Returns:
            DataFrame with features for each frame
        """
        features_list = []
        
        print("Computing features...")
        prev_img = None
        for curr_path in tqdm(frame_paths):
            curr_img = cv2.imread(curr_path, cv2.IMREAD_GRAYSCALE)

            if curr_img is None:
                continue

            features = {
                'frame_path': curr_path,
                'mean_intensity': np.mean(curr_img),
                'std_intensity': np.std(curr_img),
                'edge_density': self._compute_edge_density(curr_img),
            }

            if prev_img is not None:
                features.update({
                    'histogram_diff': self._histogram_diff(prev_img, curr_img),
                    'ssim': self._compute_ssim(prev_img, curr_img),
                    'edge_diff': self._edge_difference(prev_img, curr_img),
                })
            else:
                features.update({
                    'histogram_diff': 0.0,
                    'ssim': 1.0,
                    'edge_diff': 0.0
                })

            features_list.append(features)
            prev_img = curr_img
        
        return pd.DataFrame(features_list)
    
    def _histogram_diff(self, img1, img2):
        """Bhattacharyya distance"""
        hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    
    def _compute_ssim(self, img1, img2):
        """Structural Similarity Index"""
        # Resize if needed
        if img1.shape != img2.shape:
            h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (w, h))
            img2 = cv2.resize(img2, (w, h))
        
        # determine data_range for SSIM (avoid ambiguous default)
        try:
            data_min = min(int(img1.min()), int(img2.min()))
            data_max = max(int(img1.max()), int(img2.max()))
            data_range = data_max - data_min
        except Exception:
            data_range = 255

        if data_range == 0:
            data_range = 255

        return ssim(img1, img2, data_range=data_range)
    
    def _edge_difference(self, img1, img2):
        """Mean absolute difference of Canny edges"""
        edges1 = cv2.Canny(img1, 50, 150)
        edges2 = cv2.Canny(img2, 50, 150)
        
        if edges1.shape != edges2.shape:
            h, w = min(edges1.shape[0], edges2.shape[0]), min(edges1.shape[1], edges2.shape[1])
            edges1 = cv2.resize(edges1, (w, h))
            edges2 = cv2.resize(edges2, (w, h))
        
        diff = np.abs(edges1.astype(float) - edges2.astype(float))
        return np.mean(diff)
    
    def _compute_edge_density(self, img):
        """Fraction of edge pixels"""
        edges = cv2.Canny(img, 50, 150)
        return np.count_nonzero(edges) / edges.size


def main():
    """Example usage with CLI.

    Supports processing a single video or all videos in a directory.
    """

    parser = argparse.ArgumentParser(description='Lecture frame extractor and feature computation')
    parser.add_argument('--video', '-v', default='data/raw_videos/lecture01.mp4',
                        help='Path to a video file or a directory containing videos')
    parser.add_argument('--fps', type=float, default=1.0, help='Base frames per second to extract')
    parser.add_argument('--dense-threshold', type=float, default=0.3, help='Histogram diff threshold for dense sampling')
    parser.add_argument('--output', '-o', default='data', help='Base output directory (frames and annotations will be created inside)')
    parser.add_argument('--no-features', action='store_true', help='Only extract frames, skip computing features')
    parser.add_argument('--resize', help='Resize saved frames, format WxH (e.g. 640x360)', default=None)
    parser.add_argument('--color-mode', choices=['color', 'gray'], default='color', help='Saved image color mode')
    parser.add_argument('--neg-ratio', type=float, default=1.0, help='Number of negatives per positive to sample (balanced dataset)')
    parser.add_argument('--train-split', type=float, default=0.7, help='Fraction of videos for training')
    parser.add_argument('--val-split', type=float, default=0.15, help='Fraction of videos for validation')
    parser.add_argument('--test-split', type=float, default=0.15, help='Fraction of videos for testing')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed for splits and sampling')
    parser.add_argument('--dataset-version', type=str, default='0.1', help='Dataset version string')
    parser.add_argument('--label-scheme', choices=['is_transition', 'pairwise', 'temporal'], default='is_transition', help='Labeling scheme for dataset')
    parser.add_argument('--temporal-window', type=int, default=2, help='Window size (±k) for temporal labeling')
    parser.add_argument('--export-format', choices=['none', 'npz', 'tfrecord'], default='none', help='Export dataset for training')

    args = parser.parse_args()

    base_output = Path(args.output)
    frames_output = base_output / 'frames'
    annotations_dir = base_output / 'annotations'
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # parse resize
    resize_tuple = None
    if args.resize:
        try:
            w, h = args.resize.lower().split('x')
            resize_tuple = (int(w), int(h))
        except Exception:
            print('Invalid --resize format, expected WxH (e.g. 640x360)')
            sys.exit(1)

    extractor = LectureFrameExtractor(output_dir=str(frames_output), resize=resize_tuple, color_mode=args.color_mode)

    video_path = Path(args.video)
    all_frames = []

    # Helper: video file extensions we consider
    video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI')

    if video_path.is_dir():
        for p in sorted(video_path.iterdir()):
            if p.suffix in video_exts and p.is_file():
                df = extractor.extract_frames(p, fps=args.fps, dense_threshold=args.dense_threshold)
                if not df.empty:
                    all_frames.append(df)
    else:
        if not video_path.exists():
            print(f"Video path does not exist: {video_path}")
            sys.exit(1)
        df = extractor.extract_frames(video_path, fps=args.fps, dense_threshold=args.dense_threshold)
        if not df.empty:
            all_frames.append(df)

    if all_frames:
        frames_df = pd.concat(all_frames, ignore_index=True)
    else:
        frames_df = pd.DataFrame()

    frames_df.to_csv(annotations_dir / 'frames_metadata.csv', index=False)
    print(f"Saved frame metadata: {len(frames_df)} frames -> {annotations_dir / 'frames_metadata.csv'}")

    if frames_df.empty or args.no_features:
        if frames_df.empty:
            print('No frames extracted; skipping feature computation.')
        else:
            print('Feature computation skipped by flag.')
        return

    feature_extractor = FeatureExtractor()
    features_df = feature_extractor.compute_features(frames_df['frame_path'].tolist())

    features_df.to_csv(annotations_dir / 'frame_features.csv', index=False)
    print(f"Saved features: {len(features_df)} frames -> {annotations_dir / 'frame_features.csv'}")

    # Display sample
    print("\nSample features:")
    print(features_df.head())
    print("\nFeature statistics:")
    print(features_df.describe())

    # ----------------- Build annotation manifest for ML -----------------
    print('\nBuilding annotation manifest...')
    manifest = frames_df.copy()
    if manifest.empty:
        print('No frames to annotate; exiting.')
        return

    # rename fields to match requested schema
    manifest = manifest.rename(columns={'video_name': 'video_id'})

    # Ensure ordering by video and timestamp
    manifest = manifest.sort_values(['video_id', 'timestamp']).reset_index(drop=True)

    # add prev/next frame paths
    manifest['prev_frame_path'] = manifest.groupby('video_id')['frame_path'].shift(1)
    manifest['next_frame_path'] = manifest.groupby('video_id')['frame_path'].shift(-1)

    # label: use 'is_transition' (True when histogram diff triggered dense sampling)
    manifest['label'] = manifest.get('is_transition', False).astype(int)

    # assign slide_id: increment on each positive label within a video
    manifest['slide_id'] = 0
    for vid, grp in manifest.groupby('video_id'):
        sid = 0
        sids = []
        for lab in grp['label'].tolist():
            if int(lab) == 1:
                sid += 1
                sids.append(sid)
            else:
                sids.append(sid)
        manifest.loc[grp.index, 'slide_id'] = sids

    # MD5s were computed during frame write and stored in 'md5' column

    # include original video path column from extraction metadata
    manifest['video_path'] = manifest.get('source_video', '')

    # Apply labeling scheme
    label_scheme = args.label_scheme
    if label_scheme == 'pairwise':
        # build pairwise rows (prev_path, curr_path) with label = 1 if curr is transition
        pairs = []
        for vid, grp in manifest.groupby('video_id'):
            grp = grp.reset_index(drop=True)
            for i in range(1, len(grp)):
                prev = grp.loc[i-1]
                curr = grp.loc[i]
                pairs.append({
                    'video_id': vid,
                    'prev_frame_path': prev['frame_path'],
                    'frame_path': curr['frame_path'],
                    'frame_idx': curr['frame_idx'],
                    'timestamp': curr['timestamp'],
                    'label': int(curr.get('is_transition', False)),
                    'slide_id': int(curr.get('slide_id', 0)),
                    'md5': curr.get('md5', ''),
                    'video_path': curr.get('source_video', '')
                })
        sampled_manifest = pd.DataFrame(pairs)
    elif label_scheme == 'temporal':
        k = max(0, int(args.temporal_window))
        mf = manifest.copy()
        mf = mf.reset_index(drop=True)
        labels = []
        for idx in range(len(mf)):
            low = max(0, idx - k)
            high = min(len(mf) - 1, idx + k)
            window = mf.loc[low:high]
            # positive if any frame in window has is_transition True
            lab = int(window['is_transition'].any())
            labels.append(lab)
        mf['label'] = labels
        sampled_manifest = mf
    else:
        # default: use extractor's is_transition flag
        sampled_manifest = manifest.copy()

    # Now perform negative sampling to balance dataset (applies to pairwise or frame manifest)
    pos = sampled_manifest[sampled_manifest['label'] == 1]
    neg = sampled_manifest[sampled_manifest['label'] == 0]
    rnd = random.Random(args.seed)
    if len(pos) == 0:
        # nothing positive: keep all
        sampled_manifest = sampled_manifest.copy()
    else:
        target_neg = min(len(neg), int(len(pos) * args.neg_ratio))
        sampled_neg_idx = rnd.sample(list(neg.index), target_neg) if target_neg > 0 else []
        sampled_manifest = pd.concat([pos, neg.loc[sampled_neg_idx]], ignore_index=True).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # add a stable sample id
    sampled_manifest['sample_id'] = sampled_manifest.index.astype(str)

    # save manifest
    sampled_manifest.to_csv(annotations_dir / 'annotation_manifest.csv', index=False)
    print(f"Saved annotation manifest: {annotations_dir / 'annotation_manifest.csv'} ({len(sampled_manifest)} samples)")

    # create reproducible video-level splits
    videos = manifest['video_id'].unique().tolist()
    rnd.shuffle(videos)
    n = len(videos)
    t = int(n * args.train_split)
    v = int(n * args.val_split)
    # Ensure at least one video in train when possible for tiny datasets
    if n > 0 and t == 0 and v == 0:
        t = 1
    # Clamp values
    if t > n:
        t = n
    if t + v > n:
        v = max(0, n - t)
    train_v = videos[:t]
    val_v = videos[t:t+v]
    test_v = videos[t+v:]

    splits = {
        'train': sampled_manifest[sampled_manifest['video_id'].isin(train_v)],
        'val': sampled_manifest[sampled_manifest['video_id'].isin(val_v)],
        'test': sampled_manifest[sampled_manifest['video_id'].isin(test_v)],
    }

    for name, df_split in splits.items():
        df_split.to_csv(annotations_dir / f'{name}_manifest.csv', index=False)
        # write sample id list
        with open(annotations_dir / f'{name}_split.txt', 'w') as fh:
            for sid in df_split['sample_id'].tolist():
                fh.write(f"{sid}\n")
        print(f"Saved {name} split: {len(df_split)} samples -> {annotations_dir / f'{name}_manifest.csv'}")

    # save dataset metadata
    video_meta = []
    for vid, grp in manifest.groupby('video_id'):
        row = grp.iloc[0]
        fps = row.get('video_fps', None)
        total = row.get('video_total_frames', None)
        duration = None
        try:
            if fps and total:
                duration = float(total) / float(fps)
        except Exception:
            duration = None
        video_meta.append({'video_id': vid, 'fps': fps, 'total_frames': total, 'duration_s': duration})

    # Normalize ints to native Python types before JSON serialization
    dataset_info = {
        'version': args.dataset_version,
        'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'seed': int(args.seed),
        'num_samples': int(len(sampled_manifest)),
        'num_positives': int(len(sampled_manifest[sampled_manifest['label'] == 1])),
        'num_negatives': int(len(sampled_manifest[sampled_manifest['label'] == 0])),
        'videos': []
    }
    # Convert video_meta entries to JSON-serializable types
    for vm in video_meta:
        vm_entry = {
            'video_id': vm.get('video_id'),
            'fps': float(vm['fps']) if vm.get('fps') is not None else None,
            'total_frames': int(vm['total_frames']) if vm.get('total_frames') is not None else None,
            'duration_s': float(vm['duration_s']) if vm.get('duration_s') is not None else None,
        }
        dataset_info['videos'].append(vm_entry)

    with open(annotations_dir / 'dataset_metadata.json', 'w') as fh:
        json.dump(dataset_info, fh, indent=2)
    print(f"Saved dataset metadata -> {annotations_dir / 'dataset_metadata.json'}")

    # ----------------- Export dataset for training -----------------
    if args.export_format and args.export_format != 'none':
        print(f"Exporting dataset format: {args.export_format} ...")

        def load_image_as_array(path, gray_only=False):
            try:
                if gray_only or args.color_mode == 'gray':
                    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if im is None:
                        return None
                    return im.astype(np.uint8)
                else:
                    im = cv2.imread(path, cv2.IMREAD_COLOR)
                    if im is None:
                        return None
                    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.uint8)
            except Exception:
                return None

        if args.export_format == 'npz':
            imgs_a = []
            imgs_b = []
            labels = []
            for _, row in sampled_manifest.iterrows():
                if 'prev_frame_path' in row and pd.notna(row.get('prev_frame_path')):
                    a = load_image_as_array(row['prev_frame_path'])
                    b = load_image_as_array(row['frame_path'])
                    if a is None or b is None:
                        continue
                    imgs_a.append(a)
                    imgs_b.append(b)
                    labels.append(int(row['label']))
                else:
                    a = load_image_as_array(row['frame_path'])
                    if a is None:
                        continue
                    imgs_a.append(a)
                    labels.append(int(row['label']))

            # Convert to arrays (may be large)
            try:
                if imgs_b:
                    # pairwise arrays
                    X1 = np.stack([cv2.resize(i, (i.shape[1], i.shape[0])) for i in imgs_a])
                    X2 = np.stack([cv2.resize(i, (i.shape[1], i.shape[0])) for i in imgs_b])
                    y = np.array(labels, dtype=np.uint8)
                    np.savez_compressed(annotations_dir / 'dataset_pair.npz', X1=X1, X2=X2, y=y)
                    print(f"Saved NPZ pair dataset -> {annotations_dir / 'dataset_pair.npz'}")
                else:
                    X = np.stack([cv2.resize(i, (i.shape[1], i.shape[0])) for i in imgs_a])
                    y = np.array(labels, dtype=np.uint8)
                    np.savez_compressed(annotations_dir / 'dataset.npz', X=X, y=y)
                    print(f"Saved NPZ dataset -> {annotations_dir / 'dataset.npz'}")
            except Exception as e:
                print(f"Failed to save NPZ: {e}")

        elif args.export_format == 'tfrecord':
            try:
                import tensorflow as tf

                def _bytes_feature(value):
                    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

                def _int_feature(value):
                    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

                tf_path = annotations_dir / 'dataset.tfrecord'
                with tf.io.TFRecordWriter(str(tf_path)) as writer:
                    for _, row in sampled_manifest.iterrows():
                        if 'prev_frame_path' in row and pd.notna(row.get('prev_frame_path')):
                            with open(row['prev_frame_path'], 'rb') as fa:
                                a_bytes = fa.read()
                            with open(row['frame_path'], 'rb') as fb:
                                b_bytes = fb.read()
                            features = {
                                'image_a': _bytes_feature(a_bytes),
                                'image_b': _bytes_feature(b_bytes),
                                'label': _int_feature(int(row['label'])),
                                'sample_id': _bytes_feature(row['sample_id'].encode())
                            }
                        else:
                            with open(row['frame_path'], 'rb') as fb:
                                b_bytes = fb.read()
                            features = {
                                'image': _bytes_feature(b_bytes),
                                'label': _int_feature(int(row['label'])),
                                'sample_id': _bytes_feature(row['sample_id'].encode())
                            }
                        example = tf.train.Example(features=tf.train.Features(feature=features))
                        writer.write(example.SerializeToString())
                print(f"Saved TFRecord dataset -> {tf_path}")
            except Exception as e:
                print(f"TFRecord export failed (tensorflow may be missing): {e}")


if __name__ == '__main__':
    main()