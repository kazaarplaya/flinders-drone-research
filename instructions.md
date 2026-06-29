# Instructions

## Optional: extract audio from a YouTube video to WAV

This step is optional and not required for the framework itself. Use it only if
you need to create a local WAV from an online source.

Install the tools first:

```powershell
python -m pip install yt-dlp
```

Install `ffmpeg` separately if it is not already available on your machine.

Then run:

```powershell
yt-dlp -f bestaudio -x --audio-format wav -o "%(id)s.%(ext)s" "YOUTUBE_URL"
```

This command downloads the best available audio stream and converts it to WAV.
It does not download MP3 unless you explicitly change `--audio-format`.

## Run the audio cleaner

```powershell
.\.venv\Scripts\python.exe .\src\drone_isolation\drone_cleaner.py .\tests\DJI_M400(C3).wav --output .\data\processed\cleaned_example.wav
```

## Estimate SNR profiles

```powershell
.\.venv\Scripts\python.exe .\src\snr_analysis\snr_analysis.py --base-dir .\data\raw\snr_reference --out-dir .\data\processed\snr_profiles --plot-full --smooth-window 5
```

## Generate one synthetic example

```powershell
.\.venv\Scripts\python.exe .\src\generate.py --drone .\tests\DJI_M400(C3).wav --background .\data\raw\backgrounds\park_noise.wav --out-dir .\data\processed\single_run --phase hover --name hover_generated --smooth 5 --snr-offset -6
```

## Validate a reconstruction

```powershell
.\.venv\Scripts\python.exe .\evaluation\evaluate.py --ground-truth .\tests\DJI_M400(C3).wav --background .\data\raw\backgrounds\park_noise.wav --out-dir .\evaluation\output --phase hover --snr-offset -6 --smooth 5
```

## Compare two clips directly

```powershell
.\.venv\Scripts\python.exe .\evaluation\compare.py .\data\processed\mixes\synthetic_hover.wav .\tests\DJI_M400(C3).wav
```

## Important caveat when comparing clips

Comparison results are only meaningful when the two clips are matched fairly.

Good comparisons usually mean:
- same drone phase or similar behavior
- same or very similar scene conditions
- similar background characteristics
- same purpose of comparison, for example synthetic-vs-real scene audio rather than scene audio vs isolated audio

In this repo, the biggest caveat is:
- a poor comparison score can reflect scene mismatch rather than framework failure

Examples of unfair comparisons:
- hover vs takeoff
- park-background reconstruction vs real urban recording
- full drone-in-scene clip vs heavily cleaned isolated drone clip

For `evaluation/evaluate.py`, the background clip should be the real no-drone background from the same original scene as the ground-truth recording. Otherwise MFCC or log-spectral scores can look poor even when the pipeline is behaving as expected.
