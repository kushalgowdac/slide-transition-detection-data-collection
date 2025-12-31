# Slide Transition Dataset Creation — Handover Guide

This document explains how to use the dataset creation pipeline in this repository, what the outputs are, and the minimal checks your friend should run before producing a dataset from lecture videos.

**File structure**
project/
     ├── data/
     │   ├── raw_videos/
     │   ├── frames/
     │   └── annotations/
     ├── src/
     │   ├── extraction.py
     │   ├── features.py
     │   └── classifier.py
     ├── experiments/
     └── outputs/

**Quick summary:** the main script is [main.py](main.py). It extracts frames, marks transitions (multiple label schemes supported), computes MD5s on write, builds a manifest (`annotation_manifest.csv`), creates reproducible video-level splits, and can export NPZ or TFRecord training datasets.

**Files you should know**
- **Runner:** [main.py](main.py)
- **Outputs (default under `--output`):** `frames/` (saved frames), `annotations/annotation_manifest.csv`, `annotations/{train,val,test}_manifest.csv`, `annotations/{train,val,test}_split.txt`, `annotations/dataset_metadata.json`, optional `dataset.npz` / `dataset_pair.npz` / `dataset.tfrecord`.
- **Requirements:** [requirements.txt](requirements.txt)

**Prerequisites (on the machine that will create the dataset)**
- Python 3.8+ (3.10 recommended)
- Install Python dependencies:

```bash
pip install -r "d:/College_Life/projects/slide transition detection - data collection/requirements.txt"
```

- If you will use OCR, install Tesseract (suggested: UB-Mannheim Windows build). Note the install path (e.g. `D:\Tesseract-OCR\tesseract.exe`). Set `TESSERACT_CMD` in code or environment if required.

**Recommended disk and runtime considerations**
- Videos and extracted frames can use many GBs. Ensure destination disk has enough space.
- For large numbers of videos, prefer `--resize` to limit frame resolution and use `--export-format tfrecord` with sharding (TFRecord option available but may need TensorFlow installed).

**How to run — basic**

Extract frames + build manifest (default behavior):

```bash
python "d:/College_Life/projects/slide transition detection - data collection/main.py" --video "path/to/video_or_dir" --output data
```

Important CLI options (short reference)
- `--video, -v`: path to single video or directory of videos
- `--fps`: base frames-per-second to sample (default 1.0)
- `--dense-threshold`: histogram diff threshold to trigger dense sampling
- `--resize`: e.g. `640x360` — enforces saved frame resolution (recommended)
- `--color-mode`: `color` or `gray` (affects saved frames and export)
- `--label-scheme`: `is_transition` (default), `pairwise`, or `temporal`
  - `is_transition`: flag set when extractor detects a histogram change (frame-level positive)
  - `pairwise`: creates pairs (prev→curr) labeled positive when `curr` is a transition
  - `temporal`: mark a frame positive if any transition occurs within ±`--temporal-window` frames
- `--neg-ratio`: negatives per positive to keep (default `1.0`) — balances dataset
- `--seed`: reproducible splits/sampling
- `--export-format`: `none`, `npz`, or `tfrecord` (optional exports for training)

Example (resize + pairwise + export npz):

```bash
python main.py --video data/raw_videos --output data --resize 640x360 --color-mode gray --label-scheme pairwise --neg-ratio 1.0 --export-format npz
```

**What the script produces and manifest schema**
- `annotations/annotation_manifest.csv` — contains the dataset samples used for training/export. Columns include:
  - `sample_id`: stable sample identifier
  - `video_id`: video stem name
  - `frame_idx`, `timestamp`
  - `frame_path`, `prev_frame_path`, `next_frame_path`
  - `label` (0 or 1)
  - `slide_id` (incremental per positive event)
  - `md5` (checksum computed while writing frame)
  - `video_path` (original video path)

- Splits: `train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`, and corresponding `*_split.txt` listing sample IDs.
- `dataset_metadata.json` contains dataset-level metadata (version, created_at, seed, counts, per-video fps/duration).
- Optional exports:
  - `dataset.npz` or `dataset_pair.npz` (NPZ compressed arrays)
  - `dataset.tfrecord` (TFRecord) — requires TensorFlow to be installed

**Labeling details and conventions**
- `is_transition` labeling identifies frames saved due to histogram change; it's the default. Use `--label-scheme pairwise` when training models that take two-frame inputs (prev, curr). Use `--label-scheme temporal` when you want context-aware positive labels.
- `slide_id` increases by 1 every time a positive label is encountered; frames between positives carry the last seen slide id (0 until first positive).
- Pairwise samples: each row corresponds to (prev_frame_path, frame_path). Label = 1 if `frame_path` is a transition (current frame introduces a new slide).
- Temporal labeling: label frame positive if any `is_transition` exists in ±k frames (use `--temporal-window k`).

**Pre-handover checklist (what I recommend you do before sending to friend)**
- [ ] Confirm `requirements.txt` includes all packages you expect (OpenCV, numpy, pandas, scikit-image, tqdm).
- [ ] Ensure Tesseract instructions are included if OCR is required (path, install link).
- [ ] Run a smoke test on one small video and verify outputs:
  - Check `frames/VIDEO_NAME/` contains images and `annotations/annotation_manifest.csv` has expected columns.
  - Check `md5` is non-empty and stable across repeated runs if frames not re-extracted.
- [ ] Decide on `--resize` and `--color-mode` for the full dataset and communicate that value.
- [ ] Verify disk space and desired `--neg-ratio` for class balancing.
- [ ] (Optional) If TFRecord export is required, install TensorFlow and test `--export-format tfrecord` on the small dataset.

**Common issues & troubleshooting**
- If frames are missing or `frame_path` entries are invalid: confirm read permissions and that `cv2.imencode` succeeded.
- If `md5` is empty: check file write permissions or out-of-memory issues on huge frames.
- If TFRecord export fails: ensure `tensorflow` is installed and matches your Python version.

**Privacy & legal note**
- Confirm you have permission to process and store the lecture videos and derived images before sharing or storing them.

**If you want me to do one final thing before handover**
I can run a smoke test on a single provided sample video (extract frames, build manifest, export a small NPZ) and attach the small output manifest so your friend has a verified example. This takes one sample video path from you.

---

If this looks good I will save it as `README_for_handover.md` in the project root. Want me to also add an example small `smoke_test` command you can run locally?

## Project layout (recommended)

project/
  ├── data/
  │   ├── raw_videos/      # input videos (gitignored)
  │   ├── frames/          # extracted frames (gitignored)
  │   └── annotations/     # manifests, splits, metadata, exports
  ├── src/                 # code modules (`extraction.py`, `features.py`, `classifier.py`, `utils.py`)
  ├── experiments/         # experiment folders and logs
  ├── outputs/             # model checkpoints, exported datasets
  ├── scripts/             # helper scripts (smoke_test.ps1, prepare_data.sh)
  ├── configs/             # default YAML/JSON configs
  └── README_for_handover.md

## Quick start

1. Create a virtual environment and install dependencies (Windows PowerShell example):

```powershell
python -m pip install --user uv
uv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& .\.venv\Scripts\Activate
uv pip install -r requirements.txt
```

2. Prepare data folders:

```bash
.\scripts\prepare_data.sh
```

3. Run the smoke test (edit the sample video path):

```powershell
.\scripts\smoke_test.ps1 -VideoPath "data/raw_videos/input_video_1.mp4" -Output data -Resize 640x360
```

## Safe / non-destructive runs

Re-running the script with the same `--output` will overwrite frames and manifests by default. To avoid accidental overwrite:

- Use a timestamped output directory, e.g. in PowerShell:

```powershell
$t = Get-Date -Format yyyyMMdd_HHmmss
python main.py --video data/raw_videos --output "data_$t" --resize 640x360
```

- Or manually move the previous `data/annotations` and `data/frames` to a backup folder before re-running.

## Smoke test and verification

After a successful run, verify:
- `data/frames/<video_name>/` contains images
- `data/annotations/annotation_manifest.csv` exists and has columns described above
- `data/annotations/dataset_metadata.json` contains dataset counts and metadata

## Push to GitHub

Create an empty repository on GitHub first (e.g., `slide-transition-detection-data-collection`). Then run these commands locally from the project root to push your code:

```bash
git init
git add .
git commit -m "initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/slide-transition-detection-data-collection.git
git push -u origin main
```

Notes:
- Replace `<your-username>` with your GitHub username.
- Ensure `.gitignore` excludes large data (we included `data/frames/`, `outputs/`, etc.).
- If this is the first push, you may need to set up a personal access token for HTTPS pushes or configure SSH keys.

---

If you want, I can also create a small `README.md` badge or a GitHub Actions workflow to run the smoke test on push.