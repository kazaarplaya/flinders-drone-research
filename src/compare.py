import librosa
import numpy as np
import sys

def band_energy_ratio(file):
    y, sr = librosa.load(file, sr=None)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    band = (freqs >= 60) & (freqs <= 4000)

    return np.sum(S[band]) / np.sum(S)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare.py file1.wav file2.wav")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    for f in [file1, file2]:
        try:
            ratio = band_energy_ratio(f)
            print(f"{f}: {ratio:.4f}")
        except Exception as e:
            print(f"Error with {f}: {e}")