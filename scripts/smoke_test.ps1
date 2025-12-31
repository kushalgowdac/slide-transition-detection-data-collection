# Smoke test PowerShell script
# Run this from project root
param(
    [string]$VideoPath = "data/raw_videos/sample.mp4",
    [string]$Output = "data",
    [string]$Resize = "640x360"
)

python .\main.py --video $VideoPath --output $Output --resize $Resize --color-mode gray --label-scheme is_transition --neg-ratio 1.0

Write-Output "Smoke test finished. Check $Output/annotations/annotation_manifest.csv"
