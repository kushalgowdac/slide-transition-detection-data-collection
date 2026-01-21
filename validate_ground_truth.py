"""
Validate ground truth timestamps against model predictions
Compares manual timestamps with detected transitions
"""
import json
from pathlib import Path
import pandas as pd
import argparse

def load_ground_truth(video_id):
    """Load ground truth timestamps from file.
    
    Supports two formats:
    1. Enhanced: transition_time | ideal_frame_time
    2. Simple: transition_time only
    
    Returns: (transitions, ideal_frames)
    """
    gt_file = Path('data/ground_truth') / video_id / 'transitions.txt'
    
    if not gt_file.exists():
        return None, None
    
    transitions = []
    ideal_frames = []
    
    with open(gt_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            try:
                # Check for enhanced format: transition | ideal_frame
                if '|' in line:
                    parts = line.split('|')
                    transition_time = float(parts[0].strip())
                    ideal_time = float(parts[1].strip())
                    transitions.append(transition_time)
                    ideal_frames.append(ideal_time)
                else:
                    # Simple format: just transition time
                    transitions.append(float(line))
            except ValueError:
                continue
    
    return sorted(transitions), sorted(ideal_frames) if ideal_frames else None

def load_model_predictions(output_dir):
    """Load model predictions from frames_metadata.csv and best_slides.csv.
    
    Returns: (transitions, best_frames_df)
    """
    metadata_file = output_dir / 'annotations' / 'frames_metadata.csv'
    best_slides_file = output_dir / 'annotations' / 'best_slides.csv'
    
    if not metadata_file.exists():
        return None, None
    
    df = pd.read_csv(metadata_file)
    
    # Get transition frames
    transitions = df[df['is_transition'] == True]['timestamp'].tolist()
    
    # Load best slides if available
    best_frames_df = None
    if best_slides_file.exists():
        best_frames_df = pd.read_csv(best_slides_file)
    
    return sorted(transitions), best_frames_df

def compare_timestamps(ground_truth, predictions, tolerance=5.0):
    """
    Compare ground truth vs predictions with tolerance window.
    Returns matched, missed, false_positives
    """
    matched = []
    missed = []
    false_positives = list(predictions)  # Start with all predictions as false positives
    
    for gt_time in ground_truth:
        # Find closest prediction within tolerance
        best_match = None
        best_diff = float('inf')
        
        for pred_time in predictions:
            diff = abs(gt_time - pred_time)
            if diff <= tolerance and diff < best_diff:
                best_match = pred_time
                best_diff = diff
        
        if best_match is not None:
            matched.append((gt_time, best_match, best_diff))
            if best_match in false_positives:
                false_positives.remove(best_match)
        else:
            missed.append(gt_time)
    
    return matched, missed, false_positives

def main():
    parser = argparse.ArgumentParser(description='Validate ground truth vs model predictions')
    parser.add_argument('--tolerance', type=float, default=5.0, help='Matching tolerance in seconds')
    parser.add_argument('--video', type=str, help='Specific video to validate (optional)')
    args = parser.parse_args()
    
    gt_dir = Path('data/ground_truth')
    processed_dir = Path('data')
    
    # Get all videos with ground truth
    if args.video:
        video_ids = [args.video]
    else:
        video_ids = [d.name for d in gt_dir.iterdir() if d.is_dir()]
    
    print("="*70)
    print("GROUND TRUTH VALIDATION")
    print("="*70)
    print(f"Tolerance: ±{args.tolerance}s")
    print()
    
    results = []
    
    for video_id in sorted(video_ids):
        gt_transitions, gt_ideal_frames = load_ground_truth(video_id)
        
        if gt_transitions is None or len(gt_transitions) == 0:
            print(f"\n{video_id}:")
            print("  ⚠ No ground truth found")
            continue
        
        # Find corresponding output directory
        output_dir = processed_dir / f'processed_{video_id}'
        
        if not output_dir.exists():
            print(f"\n{video_id}:")
            print(f"  ⚠ Not processed yet (no output at {output_dir})")
            continue
        
        predictions, best_frames_df = load_model_predictions(output_dir)
        
        if predictions is None:
            print(f"\n{video_id}:")
            print("  ⚠ No predictions found")
            continue
        
        # Compare transitions
        matched, missed, false_positives = compare_timestamps(
            gt_transitions, predictions, args.tolerance
        )
        
        total_gt = len(gt_transitions)
        total_pred = len(predictions)
        accuracy = len(matched) / total_gt * 100 if total_gt > 0 else 0
        precision = len(matched) / total_pred * 100 if total_pred > 0 else 0
        
        print(f"\n{video_id}:")
        print(f"  Ground Truth: {total_gt} transitions")
        print(f"  Predictions:  {total_pred} transitions")
        print(f"  Matched:      {len(matched)} ({accuracy:.1f}% recall)")
        print(f"  Missed:       {len(missed)}")
        print(f"  False Pos:    {len(false_positives)}")
        print(f"  Precision:    {precision:.1f}%")
        
        # Validate ideal frames if available
        if gt_ideal_frames and best_frames_df is not None:
            print(f"\n  Ideal Frame Validation:")
            ideal_matched = 0
            for gt_ideal in gt_ideal_frames:
                # Get model's top-ranked frame for this region
                nearby_frames = best_frames_df[
                    (best_frames_df['timestamp'] >= gt_ideal - args.tolerance) &
                    (best_frames_df['timestamp'] <= gt_ideal + args.tolerance) &
                    (best_frames_df['rank'] == 1)
                ]
                if len(nearby_frames) > 0:
                    ideal_matched += 1
                    model_time = nearby_frames.iloc[0]['timestamp']
                    diff = abs(gt_ideal - model_time)
                    print(f"    ✓ GT: {gt_ideal:6.1f}s → Model: {model_time:6.1f}s (Δ {diff:.1f}s)")
                else:
                    print(f"    ✗ GT: {gt_ideal:6.1f}s → No model match within ±{args.tolerance}s")
            
            ideal_accuracy = ideal_matched / len(gt_ideal_frames) * 100 if gt_ideal_frames else 0
            print(f"  Ideal Frame Match: {ideal_matched}/{len(gt_ideal_frames)} ({ideal_accuracy:.1f}%)")
        
        if matched:
            print("\n  Matched transitions:")
            for gt, pred, diff in matched:
                print(f"    GT: {gt:6.1f}s  →  Pred: {pred:6.1f}s  (Δ {diff:.1f}s)")
        
        if missed:
            print("\n  Missed transitions:")
            for gt in missed:
                print(f"    GT: {gt:6.1f}s  (no match within ±{args.tolerance}s)")
        
        if false_positives:
            print("\n  False positives:")
            for fp in false_positives:
                print(f"    Pred: {fp:6.1f}s  (no GT match)")
        
        results.append({
            'video_id': video_id,
            'ground_truth_count': total_gt,
            'prediction_count': total_pred,
            'matched': len(matched),
            'missed': len(missed),
            'false_positives': len(false_positives),
            'recall': accuracy,
            'precision': precision
        })
    
    # Summary
    if results:
        print("\n" + "="*70)
        print("OVERALL SUMMARY")
        print("="*70)
        
        avg_recall = sum(r['recall'] for r in results) / len(results)
        avg_precision = sum(r['precision'] for r in results) / len(results)
        total_matched = sum(r['matched'] for r in results)
        total_gt = sum(r['ground_truth_count'] for r in results)
        total_pred = sum(r['prediction_count'] for r in results)
        
        print(f"Videos validated: {len(results)}")
        print(f"Total ground truth transitions: {total_gt}")
        print(f"Total predicted transitions: {total_pred}")
        print(f"Total matched: {total_matched}")
        print(f"Average recall: {avg_recall:.1f}%")
        print(f"Average precision: {avg_precision:.1f}%")
        
        # Save results
        results_df = pd.DataFrame(results)
        results_file = Path('validation_results.csv')
        results_df.to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")

if __name__ == '__main__':
    main()
