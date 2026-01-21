"""
MASTER AUTOMATION SCRIPT
Runs entire pipeline with resume capability at each stage

Stages:
1. Process videos (extract frames, detect transitions)
2. Validate against ground truth
3. Create labeled dataset
4. Train ML model

Each stage is resumable - if interrupted, will continue from last completed step.
"""

import subprocess
import sys
from pathlib import Path
import time
from datetime import datetime

def run_stage(name, script, description):
    """Run a pipeline stage with error handling."""
    print("\n" + "="*70)
    print(f"STAGE: {name}")
    print("="*70)
    print(description)
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print("="*70)
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script], check=True)
        elapsed = time.time() - start_time
        print(f"\n✓ {name} completed in {elapsed/60:.1f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {name} failed after {elapsed/60:.1f} minutes")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n⚠ {name} interrupted after {elapsed/60:.1f} minutes")
        print("Progress has been saved. You can resume by running this script again.")
        return False

def check_stage_completion(stage):
    """Check if a stage is already completed."""
    checks = {
        'process': lambda: len(list(Path('data').glob('processed_*/annotations/.extraction_complete'))) >= 10,
        'validate': lambda: Path('validation_results.csv').exists(),
        'dataset': lambda: Path('labeled_dataset.csv').exists(),
        'train': lambda: Path('trained_model.pth').exists()
    }
    
    return checks.get(stage, lambda: False)()

def main():
    """Run full pipeline."""
    print("="*70)
    print("AUTOMATED SLIDE TRANSITION DETECTION PIPELINE")
    print("="*70)
    print("This will run the complete pipeline:")
    print("  1. Extract frames from all videos")
    print("  2. Validate against ground truth")
    print("  3. Create labeled training dataset")
    print("  4. Train ML classifier")
    print()
    print("⚡ RESUMABLE: If interrupted, progress is saved automatically")
    print("   Just run this script again to continue from where you left off.")
    print("="*70)
    
    response = input("\nStart pipeline? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    pipeline_start = time.time()
    
    # Stage 1: Process videos
    if check_stage_completion('process'):
        print("\n✓ STAGE 1: Video processing already complete (skipping)")
    else:
        success = run_stage(
            "STAGE 1: Video Processing",
            "process_with_ground_truth.py",
            "Extracting frames and detecting transitions from all videos..."
        )
        if not success:
            print("\n⚠ Pipeline paused at Stage 1. Run script again to resume.")
            return
    
    # Stage 2: Validation
    if check_stage_completion('validate'):
        print("\n✓ STAGE 2: Validation already complete (skipping)")
    else:
        success = run_stage(
            "STAGE 2: Validation",
            "validate_ground_truth.py",
            "Comparing model predictions vs ground truth..."
        )
        # Validation failure is not critical, continue anyway
    
    # Stage 3: Create dataset
    if check_stage_completion('dataset'):
        print("\n✓ STAGE 3: Dataset already created (skipping)")
    else:
        success = run_stage(
            "STAGE 3: Dataset Creation",
            "create_dataset.py",
            "Creating labeled training dataset from processed videos..."
        )
        if not success:
            print("\n⚠ Pipeline paused at Stage 3. Run script again to resume.")
            return
    
    # Stage 4: Train model
    if check_stage_completion('train'):
        print("\n✓ STAGE 4: Model already trained (skipping)")
    else:
        print("\n" + "="*70)
        print("STAGE 4: Model Training")
        print("="*70)
        print("This will train a neural network for ~10 epochs.")
        print("Expected time: 30-60 minutes (depending on GPU availability)")
        print("="*70)
        
        response = input("\nStart training? (y/n): ")
        if response.lower() != 'y':
            print("Training skipped. You can run train_model.py later.")
        else:
            success = run_stage(
                "STAGE 4: Model Training",
                "train_model.py",
                "Training CNN classifier on labeled dataset..."
            )
            if not success:
                print("\n⚠ Training incomplete. Run train_model.py to resume.")
                return
    
    # Pipeline complete
    pipeline_elapsed = time.time() - pipeline_start
    
    print("\n" + "="*70)
    print("🎉 PIPELINE COMPLETE!")
    print("="*70)
    print(f"Total time: {pipeline_elapsed/60:.1f} minutes")
    print()
    print("Output files:")
    print("  - validation_results.csv: Model accuracy vs ground truth")
    print("  - labeled_dataset.csv: Training dataset")
    print("  - trained_model.pth: Trained classifier")
    print("  - training_history.json: Training metrics")
    print()
    print("Next steps:")
    print("  1. Review validation_results.csv for model accuracy")
    print("  2. Check training_history.json for training progress")
    print("  3. Integrate trained_model.pth into main.py for inference")
    print("="*70)

if __name__ == '__main__':
    main()
