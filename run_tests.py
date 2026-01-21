"""
Run tests on both videos without interruption and save all results
"""

import subprocess
import json
from pathlib import Path

videos = [
    ("data/testing_videos/db_1.mp4", "data/testing_videos/db_1_transtions.txt", "db_1"),
    ("data/testing_videos/toc_1.mp4", "data/testing_videos/toc_1_transitions.txt", "toc_1"),
]

results = {}

for video_path, gt_path, video_name in videos:
    print(f"\n{'='*80}")
    print(f"TESTING: {video_name}")
    print('='*80)
    
    cmd = [
        '.venv\\Scripts\\python.exe',
        'test_model_professional.py',
        '--video', video_path,
        '--ground-truth', gt_path,
        '--fps', '1.0'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        print(f"\nExit code: {result.returncode}")
        results[video_name] = {
            'exit_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Test timed out for {video_name}")
        results[video_name] = {'error': 'timeout'}
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        results[video_name] = {'error': str(e)}

# Save results
with open('test_run_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{'='*80}")
print("Test run complete. Results saved to test_run_results.json")
print('='*80)
