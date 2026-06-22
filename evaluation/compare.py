"""Band Power Ratio (BPR) — the documentation's validation metric.

Power-based (amplitude squared), restricted to the drone harmonic band
150-6000 Hz. This range best separates drone energy from outdoor background
(low-frequency wind/ambient is excluded; the upper prop-harmonic comb is
kept). A higher BPR means more of the signal's energy sits in the drone band.
"""

import argparse
import sys

import librosa
import numpy as np


def band_power_ratio(file_path, low=150, high=6000, sr=22050):
    """Fraction of signal POWER inside [low, high] Hz.

    Audio is resampled to a fixed sample rate so the ratio is comparable
    across files (BPR is sample-rate sensitive otherwise).
    """
    y, sr = librosa.load(file_path, sr=sr)
    power = np.abs(librosa.stft(y)) ** 2
    freqs = librosa.fft_frequencies(sr=sr)

    band = (freqs >= low) & (freqs <= high)
    return float(np.sum(power[band]) / (np.sum(power) + 1e-12))


def mfcc_distance(file_a, file_b, sr=22050, n_mfcc=20):
    """Timbre distance: cosine distance between time-averaged MFCC vectors.

    Coefficient 0 (overall log-energy / level) is dropped so this measures
    timbre *shape*, not loudness -- loudness is already captured by BPR/SNR.
    Cosine distance is scale-invariant. 0 = identical timbre.
    """
    a, _ = librosa.load(file_a, sr=sr)
    b, _ = librosa.load(file_b, sr=sr)
    ma = librosa.feature.mfcc(y=a, sr=sr, n_mfcc=n_mfcc)[1:].mean(axis=1)
    mb = librosa.feature.mfcc(y=b, sr=sr, n_mfcc=n_mfcc)[1:].mean(axis=1)
    cos = np.dot(ma, mb) / (np.linalg.norm(ma) * np.linalg.norm(mb) + 1e-9)
    return float(1.0 - cos)


def log_spectral_distance(file_a, file_b, sr=22050, low=150, high=6000):
    """Spectral-shape distance: RMS dB difference over the drone band.

    Restricted to [low, high] Hz (outside it is near-silence whose dB floor is
    meaningless noise) and level-normalised (mean dB removed) so it measures the
    spectral *shape* of the drone, not its loudness. Units dB; 0 = identical.
    """
    a, _ = librosa.load(file_a, sr=sr)
    b, _ = librosa.load(file_b, sr=sr)
    sa = np.abs(librosa.stft(a)).mean(axis=1)
    sb = np.abs(librosa.stft(b)).mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr)
    n = min(len(sa), len(sb))
    band = (freqs[:n] >= low) & (freqs[:n] <= high)
    da = 20.0 * np.log10(sa[:n][band] + 1e-10)
    db = 20.0 * np.log10(sb[:n][band] + 1e-10)
    da = da - da.mean()
    db = db - db.mean()
    return float(np.sqrt(np.mean((da - db) ** 2)))


def main():
    parser = argparse.ArgumentParser(
        description="Band Power Ratio (power, 150-6000 Hz) for one or more WAVs."
    )
    parser.add_argument("files", nargs="+", help="WAV files to measure.")
    parser.add_argument("--low", type=float, default=150)
    parser.add_argument("--high", type=float, default=6000)
    args = parser.parse_args()

    for f in args.files:
        try:
            ratio = band_power_ratio(f, args.low, args.high)
            print(f"{f}")
            print(f"Band Power Ratio ({args.low:.0f}-{args.high:.0f} Hz): {ratio:.4f}\n")
        except Exception as e:
            print(f"Error processing {f}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
