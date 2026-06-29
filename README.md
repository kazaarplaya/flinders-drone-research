# Flinders Drone Research

Framework for generating synthetic drone-audio datasets, estimating drone/background
SNR over time, isolating drone signatures, mixing them onto new scenes, and
validating the results against real recordings.

## Repository Layout

```text
src/
  generate.py                  # End-to-end single-clip generation pipeline
  drone_classification/
  drone_isolation/
  drone_mixer/
  snr_analysis/

evaluation/
  evaluate.py                  # Reconstruct + compare synthetic vs real
  compare.py                   # BPR/MFCC/log-spectral comparison helpers

data/
  raw/
    backgrounds/               # Canonical background recordings
    snr_reference/             # Background + phase clips used for SNR estimation
    C1/                        # Additional raw reference audio
    C3/                        # Additional raw reference audio
  processed/
    mixes/                     # Small demo set of synthetic outputs
    snr_profiles/              # Saved SNR curves

tests/
  DJI_M400(C3).wav             # Main example real drone recording
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 1. Estimate SNR Profiles

```powershell
.\.venv\Scripts\python.exe .\src\snr_analysis\snr_analysis.py --base-dir .\data\raw\snr_reference --out-dir .\data\processed\snr_profiles --plot-full --smooth-window 5
```

## 2. Generate One Synthetic Clip

```powershell
.\.venv\Scripts\python.exe .\src\generate.py --drone .\tests\DJI_M400(C3).wav --background .\data\raw\backgrounds\park_noise.wav --out-dir .\data\processed\single_run --phase hover --name hover_generated --smooth 5 --snr-offset -6
```

To generate onto the urban background, add `--mix-onto .\data\raw\backgrounds\urban_environment.wav`.

## 3. Validate Against a Real Recording

```powershell
.\.venv\Scripts\python.exe .\evaluation\evaluate.py --ground-truth .\tests\DJI_M400(C3).wav --background .\data\raw\backgrounds\park_noise.wav --out-dir .\evaluation\output --phase hover --snr-offset -6 --smooth 5
```

Important: validation only makes sense when `--background` is the real no-drone
background from the same scene as the ground-truth recording.

## Notes

- Canonical inputs live under `data/raw/`.
- Generated outputs should go under `data/processed/`, `evaluation/output/`, or `use_cases/*/output/`.
- The repo is intentionally cleaned for handoff, so regenerate outputs instead of treating generated files as source data.
