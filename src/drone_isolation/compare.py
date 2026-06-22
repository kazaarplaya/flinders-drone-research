import librosa
import numpy as np
import sys


def band_power_ratio(file, low=60, high=4000):
    # Load audio
    y, sr = librosa.load(file, sr=None)

    # STFT
    S = np.abs(librosa.stft(y))

    # Convert magnitude -> power
    power = S ** 2

    # Frequency bins
    freqs = librosa.fft_frequencies(sr=sr)

    # Frequency mask
    band = (freqs >= low) & (freqs <= high)

    # Compute ratio
    band_power = np.sum(power[band])
    total_power = np.sum(power)

    ratio = band_power / (total_power + 1e-12)

    return ratio


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("python compare1.py file1.wav file2.wav")
        sys.exit(1)

    files = sys.argv[1:]

    for f in files:
        try:
            ratio = band_power_ratio(f)

            print(f"{f}")
            print(f"Band Power Ratio (60Hz-4000Hz): {ratio:.4f}")
            print()

        except Exception as e:
            print(f"Error processing {f}: {e}")