# Next Steps (Project Plan)

## Current status
- v3 model trained with hard positives/negatives.
- Best sweep config found: thresh=0.55, diff-pct=90, min-gap=3.0.
- Detection results saved in results_enriched_v3_best/ and summary_10s.csv.

## 1) Precision recovery (recommended first)
- Add a **post-filter** on detected transitions:
  - Keep only detections with confidence >= (threshold + 0.10).
  - Optionally enforce a stricter **per-cluster** max-confidence cutoff.
- Run a small sweep for (threshold, diff-pct, min-gap, post-filter cutoff).
- Choose operating point by mean F1 or precision@recall≥0.6.

## 2) Best-slide selection (main goal)
- For each detected transition timestamp, extract a small window (e.g., ±2s, sample every 0.2s).
- Cluster frames by visual similarity (pHash or SSIM).
- Score each frame: maximize content_fullness, minimize occlusion (skin_ratio/is_occluded) and blur (low Laplacian variance).
- Select the best frame per cluster, then output:
  - slide_timestamp (selected frame time)
  - best_frame_path
  - quality scores

## 3) Deliverables
- A JSON/CSV with best slide timestamps and file paths.
- (Optional) A folder of best frames per slide.

## 4) Evaluation
- Compare detected transitions with stricter tolerances (e.g., 2s/5s) **only for reporting**.
- Use the same tolerance for all models to compare fairly.

## 5) Optional model improvements
- Gather more labeled positives from board/handwriting videos.
- Train a separate model per style (ppt vs board), or add style classifier.
