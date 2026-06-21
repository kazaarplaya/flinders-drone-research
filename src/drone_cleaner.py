import numpy as np
import librosa
import soundfile as sf
import scipy.signal as signal
import noisereduce as nr
import sys
import os

from scipy.signal import medfilt


def rms_normalize(y, target_rms=0.08):
    rms = np.sqrt(np.mean(y**2) + 1e-12)
    return y * (target_rms / (rms + 1e-12))


def butter_highpass(y, sr, cutoff=80, order=4):
    b, a = signal.butter(order, cutoff / (sr / 2), btype="high")
    return signal.lfilter(b, a, y)


def butter_bandpass(y, sr, low=100, high=3000, order=4):
    b, a = signal.butter(order, [low / (sr / 2), high / (sr / 2)], btype="band")
    return signal.lfilter(b, a, y)

def stft_drone_mask(y, sr, n_fft=2048, hop_length=512, low=60, high=4000):
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_mag, S_phase = librosa.magphase(D)

    # Estimate per-frequencty noise floor
    noise_floor = np.percentile(S_mag, 20, axis=1, keepdims=True)

    # Soft mask
    mask = S_mag / (S_mag + noise_floor + 1e-8)

    # Frequency weighting toward likely drone band
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band_weight = np.ones((len(freqs), 1)) * 0.25
    band_weight[(freqs >= low) & (freqs <= high)] = 1.0
    mask *= band_weight

    # Smooth mask
    mask = medfilt(mask, kernel_size=(5,5))
    mask = np.clip(mask, 0.0, 1.0)

    S_clean = S_mag * mask
    y_clean = librosa.istft(S_clean * S_phase, hop_length=hop_length)

    return y_clean


def isolate_drone_audio(input_path, output_path):
    # Load as mono
    y, sr = librosa.load(input_path, sr=None, mono=True)

    # Normalize
    y = rms_normalize(y)

    # Remove low-frequency rumble/wind
    y = butter_highpass(y, sr, cutoff=80)

    # Keep likely drone-heavy range
    y = butter_bandpass(y, sr, low=60, high=4000)

    # Reduce stationary background noise
    y = nr.reduce_noise(y=y, sr=sr, stationary=True)

    # STFT masking stage
    y_clean = stft_drone_mask(y, sr, low=60, high=4000)

    # Optional HPSS after masking
    # Keep more steady/harmonic content, suppress transient junk
    # harmonic, _ = librosa.effects.hpss(y)
    # y_clean = harmonic

    # Normalize again
    y_clean = rms_normalize(y_clean)

    # Prevent clipping
    y_clean = np.clip(y_clean, -1.0, 1.0)

    # Save
    sf.write(output_path, y_clean, sr)
    print(f"Saved cleaned audio to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python drone_cleaner.py input.wav")
    else:
        input_path = sys.argv[1]

        # Split filename and extension
        base, ext = os.path.splitext(input_path)

        # Create new filename
        output_path = f"{base}(clean){ext}"

        isolate_drone_audio(input_path, output_path)