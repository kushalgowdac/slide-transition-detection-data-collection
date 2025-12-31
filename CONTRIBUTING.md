# Contributing / Handover Checklist

Small checklist for collaborators who will create datasets using this repository.

- Create or activate the virtual environment (we recommend `uv venv` on Windows).
- Install dependencies: `pip install -r requirements.txt`.
- Verify Tesseract if OCR is required.
- Choose dataset parameters: `--resize`, `--color-mode`, `--label-scheme`, `--neg-ratio`.
- Run smoke test using `scripts\smoke_test.ps1` or run the example command in `README_for_handover.md`.
- Inspect `data/annotations/annotation_manifest.csv` (columns listed in README).
- If creating multiple experiments, store logs/checkpoints under `experiments/<name>/`.
