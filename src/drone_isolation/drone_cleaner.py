import numpy as np
import librosa
import soundfile as sf
import scipy.signal as signal
import noisereduce as nr
import sys
import os


def rms_normalize(y, target_rms=0.08):
    rms = np.sqrt(np.mean(y ** 2) + 1e-12)
    return y * (target_rms / (rms + 1e-12))


def butter_highpass(y, sr, cutoff=180, order=4):
    b, a = signal.butter(order, cutoff / (sr / 2), btype="high")
    return signal.filtfilt(b, a, y)


def butter_bandpass(y, sr, low=180, high=7000, order=4):
    b, a = signal.butter(
        order,
        [low / (sr / 2), high / (sr / 2)],
        btype="band"
    )

    return signal.filtfilt(b, a, y)


def spectral_gate(
    y,
    sr,
    n_fft=2048,
    hop_length=512,
    noise_percentile=15
):
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    magnitude = np.abs(D)
    phase = np.angle(D)

    # Estimate noise floor per frequency bin
    noise_floor = np.percentile(
        magnitude,
        noise_percentile,
        axis=1,
        keepdims=True
    )

    # Soft attenuation instead of hard masking
    attenuation = np.maximum(
        magnitude - noise_floor,
        0
    ) / (magnitude + 1e-10)

    # Preserve some ambience to avoid artifacts
    attenuation = 0.25 + (0.75 * attenuation)

    cleaned_mag = magnitude * attenuation

    cleaned = librosa.istft(
        cleaned_mag * np.exp(1j * phase),
        hop_length=hop_length
    )

    return cleaned


def harmonic_enhancement(y):
    harmonic, percussive = librosa.effects.hpss(y)

    # Keep mostly harmonic content
    return (0.85 * harmonic) + (0.15 * percussive)


def isolate_drone_audio(input_path, output_path):

    # Load mono audio
    y, sr = librosa.load(
        input_path,
        sr=None,
        mono=True
    )

    # Normalize
    y = rms_normalize(y)

    # Remove wind rumble
    y = butter_highpass(
        y,
        sr,
        cutoff=180
    )

    # Focus on drone harmonic range
    y = butter_bandpass(
        y,
        sr,
        low=180,
        high=7000
    )

    # Mild adaptive noise reduction
    y = nr.reduce_noise(
        y=y,
        sr=sr,
        stationary=False,
        prop_decrease=0.25
    )

    # Spectral gating
    y = spectral_gate(y, sr)

    # Emphasize harmonic drone structure
    y = harmonic_enhancement(y)

    # Final normalize
    y = rms_normalize(y)

    # Prevent clipping
    y = np.clip(y, -1.0, 1.0)

    # Save
    sf.write(output_path, y, sr)

    print(f"Saved cleaned audio to: {output_path}")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python drone_isolation.py input.wav")

    else:
        input_path = sys.argv[1]

        base, ext = os.path.splitext(input_path)

        output_path = f"{base}(isolated){ext}"

        isolate_drone_audio(
            input_path,
            output_path
        )