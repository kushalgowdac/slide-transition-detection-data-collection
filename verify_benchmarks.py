"""
Proper benchmark comparison script
Ask user to provide actual timestamps for benchmark slides
"""
import pandas as pd
from pathlib import Path

# Load frames metadata
all_frames = pd.read_csv('data/annotations/frames_metadata.csv')

print("\n" + "="*80)
print("BENCHMARK VERIFICATION HELPER")
print("="*80)

print("""
I found your benchmark images but the filenames don't have clear timestamps:

VIDEO 3:
  slide_000.jpg to slide_009.jpg (10 transitions)
  Assumption: 1 second apart? 0s, 1s, 2s, ... 9s?
  Please confirm: Are these actual timestamps OR just sequential numbering?

VIDEO 4:
  t0.0.png, t0.38.png, t0.39.png, ... t4.41.png (18 transitions)
  These look like minutes (t0.0 = 0.0 min = 0s, t0.38 = 0.38 min = 23s, etc.)
  
QUESTION:
For proper accuracy comparison, I need the ACTUAL TIMESTAMP (in seconds) 
where you marked each transition.

Can you provide:
1. For video_3: List of 10 timestamps (seconds)
2. For video_4: List of 18 timestamps (seconds) from the filenames

For video_4, the filenames suggest:
  t0.0 = 0.0 min = 0s
  t0.38 = 0.38 min = 22.8s
  t0.39 = 0.39 min = 23.4s
  t0.50 = 0.50 min = 30s
  ... etc

Is this correct?
""")

print("\n" + "="*80)
print("CURRENT MODEL PERFORMANCE ON VIDEO 4:")
print("="*80)

frames_v4 = all_frames[all_frames['video_name'] == 'input_video_4'].copy()
frames_v4 = frames_v4.sort_values('timestamp').reset_index(drop=True)
frames_v4['is_transition'] = (
    (frames_v4['edge_change'] > 4.0) | 
    (frames_v4['content_fullness'].diff().abs() > 0.15)
)

model_trans_v4 = frames_v4[frames_v4['is_transition']==True]['timestamp'].tolist()

print(f"\nModel detected {len(model_trans_v4)} transitions:")
for i, t in enumerate(model_trans_v4, 1):
    mins = int(t // 60)
    secs = int(t % 60)
    print(f"  {i:2d}. {t:6.1f}s ({mins}:{secs:02d})")

print("\nOnce you provide the actual benchmark timestamps, I'll calculate:")
print("  - Exact accuracy percentage")
print("  - Which transitions were missed")
print("  - Whether thresholds need adjustment")
