# flinders-drone-research

Framework for generating **synthetic drone-audio datasets** and validating them
against real recordings. It isolates a drone's acoustic signature, measures its
loudness over time (SNR), then mixes it onto a background scene at that loudness.

## Pipeline
```
                       ┌───────────────────────┐
                       │   Drone recordings    │
                       │  • flight-phase clips │
                       └───────────┬───────────┘
             ┌─────────────────────┴───────────────────┐
             ▼                                         ▼
 ┌───────────────────┐   ┌────────────────────┐   ┌──────────────────────┐
 │  Drone isolation  │   │    SNR analysis    │   │ Background recording │
 │ • RMS normalise   │   │ • 40/20 ms frames  │   │ • outdoor target     │
 │ • find drone band │   │ • drone vs med bg  │   │   scene              │
 │ • band-pass+gate  │   │ • smoothed SNR     │   │                      │
 │ • HPSS boost      │   │   profile (JSON)   │   │                      │
 └─────────┬─────────┘   └─────────┬──────────┘   └──────────┬───────────┘
           └───────────────────────┼─────────────────────────┘
                                   ▼
                         ┌────────────────────┐
                         │  Mixing algorithm  │
                         │ • match/tile length│
                         │ • scale to target  │
                         │ • overlap-add      │
                         └─────────┬──────────┘
                                   ▼
                         ┌────────────────────┐
                         │  Synthetic dataset │
                         │ • drone + scene WAV│
                         │ • SNR-matched      │
                         └─────────┬──────────┘
                                   ▼
                         ┌────────────────────┐
                         │     Validation     │
                         │ • BPR 150-6000 Hz  │
                         │ • MFCC+log-spectral│
                         │ • similarity 0-1   │
                         └────────────────────┘
```

## Setup
```bash
python3.13 -m venv .venv
.venv/bin/pip install numpy scipy soundfile librosa noisereduce matplotlib
```
Use `.venv/bin/python` for all commands (the system Python has no deps), or
`source .venv/bin/activate` first and use `python`.

---

## 1. Generate a synthetic clip — `src/generate.py`

Isolates the drone, builds its SNR profile, and mixes it onto a background.

```bash
python src/generate.py \
  --drone      <drone.wav> \
  --background <scene_background.wav> \
  --mix-onto   <target_scene.wav> \
  --out-dir    <output_dir> \
  --phase <phase> --name <output_name> --min-low 150
```

| arg | required | default | description |
|-----|----------|---------|-------------|
| `--drone` | yes | – | real drone-in-scene WAV (isolation + SNR source) |
| `--background` | yes | – | background-only WAV from the **same** scene (SNR reference) |
| `--out-dir` | yes | – | directory for outputs |
| `--mix-onto` | no | `--background` | scene to mix the drone onto; set a *new* scene to generate new data |
| `--phase` | no | `hover` | label for outputs (takeoff, hover, altitude_increase, …) |
| `--name` | no | `--phase` | basename for the synthetic WAV |
| `--min-low` | no | `60` | highpass floor (Hz); raise to 120–150 to cut wind when mixing into a new scene |
| `--snr-offset` | no | `0` | dB shift on the profile; negative = louder background |
| `--smooth` | no | `5` | SNR profile smoothing window (frames) |

Outputs:
```
<out-dir>/isolated/<phase>.wav      isolated drone
<out-dir>/profiles/<phase>_snr.json SNR profile
<out-dir>/synthetic/<name>.wav      the synthetic clip
```
The `synthetic/` folder is cleared each run.

### Two modes
- **Generation** (drone into a new scene): set `--mix-onto` to a different
  background, and raise `--min-low` (e.g. 150) so the original scene's wind
  doesn't bleed in.
- **Faithful reconstruction**: omit `--mix-onto` — mixes back onto the original
  background. Keep `--min-low` low (60) to preserve the drone's low harmonics.

---

## 2. Validate against the real recording — `evaluation/evaluate.py`

Runs the whole pipeline, then scores the synthetic vs the real drone clip.

```bash
python evaluation/evaluate.py \
  --ground-truth <drone.wav> \
  --background   <scene_background.wav> \
  --out-dir      <output_dir> \
  --phase <phase>
```

| arg | required | default | description |
|-----|----------|---------|-------------|
| `--ground-truth` | yes | – | real drone-in-scene WAV (also the comparison target) |
| `--background` | yes | – | same-scene background |
| `--out-dir` | yes | – | directory for outputs |
| `--phase` | no | `hover` | output label |
| `--snr-offset` | no | `0` | dB shift on the profile |
| `--smooth` | no | `5` | SNR smoothing window |

Writes `<out-dir>/results/comparison.json` with three calibrated 0–1
similarities (1 = matches the real recording):

| metric | measures |
|--------|----------|
| `bpr` | energy in the drone band (150–6000 Hz) |
| `mfcc` | timbre |
| `log_spectral` | spectral shape |
| `overall` | mean of the three |

> Validation only works when mixing onto the **same scene** as the ground truth.
> A new scene (`--mix-onto`) has no real recording to compare against.

---

## 3. SNR profile only — `src/snr_analysis/snr_analysis.py`

```bash
python src/snr_analysis/snr_analysis.py \
  --base-dir <dir_with_background_and_phase_wavs> \
  --phases <phase> \
  --out-dir <output_dir>
```

| arg | default | description |
|-----|---------|-------------|
| `--base-dir` | `.` | folder with `background.wav` + `<phase>.wav` files |
| `--background` | `<base-dir>/background.wav` | explicit background path |
| `--phases` | all 5 | subset to process (e.g. `hover landing`) |
| `--out-dir` | `data/processed/snr_profiles` | where to write `<phase>_snr.json` |
| `--smooth-window` | `0` | moving-average window |
| `--plot-full` | off | plot SNR across phases |

Phases: `takeoff hover altitude_increase altitude_decrease landing`.

---

## 4. Compare any two clips — `evaluation/compare.py`

```bash
python evaluation/compare.py <synthetic.wav> <real.wav>
```
Prints the Band Power Ratio (power, 150–6000 Hz) of each file.

---

## Use-case layout
```
use_cases/<scene>/
├── drone/
│   ├── <phase>.wav        real drone in scene  (ground truth)
│   └── background.wav     same scene, no drone
├── background/
│   └── <scene>.wav        target scene for generation
└── output/
    ├── generation/        synthetic clips (new scene)
    └── validation/        synthetic + results/comparison.json
```
Only `drone/` and `background/` are real inputs — everything in `output/` is
regenerated.
