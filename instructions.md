# Instructions
## Convert YouTube video -> .wav
To download and convert a YouTube video to a .wav file, copy the command into bash in target directory.

1. install yt-dlp
  ```bash
  brew install yt-dlp
  ```
2. install ffmpeg
```bash
  brew install ffmpeg
  ```
3. Run the command:
  ```bash 
  yt-dlp -f bestaudio -x --audio-format wav -o "%(id)s.%(ext)s" "YOUTUBE_URL"
  ```
---
## Run Audio Cleaner
- cd to the directory with the *drone_cleaner.py* AND .wav files
- activate virtual environment:
```bash 
source venv/bin/activate 
```
- run the script:

```bash
python drone_cleaner.py "FILE_NAME.wav"
```

---

## Compare original and clean files
The ```compare.py``` script compares two files for their energy concentration in a given frequency range (60-4000 Hz in this instance).

Run the script:
```bash
python compare1.py "FILE_1.wav" "FILE_2.wav"
```

Example output:
```bash
file1.wav: 0.82
file2.wav: 0.98
```
This output indicates that file1 has 82% of energy inside the drone band, and file2 has 98%.
