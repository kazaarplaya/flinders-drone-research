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
    noise_percentile=15,
    low=None,
    high=None,
    out_of_band_gain=0.1
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

    # Crush bins outside the estimated drone band
    if low is not None and high is not None:
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        in_band = (freqs >= low) & (freqs <= high)
        band_gain = np.where(in_band, 1.0, out_of_band_gain)
        attenuation = attenuation * band_gain[:, None]

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


def estimate_drone_band(
    y,
    sr,
    n_fft=4096,
    hop_length=1024,
    f0_min=40,
    f0_max=400,
    n_harmonics=5,
    low_pct=0.01,
    high_pct=0.995,
    min_low=60.0
):

    # Long-term average magnitude spectrum
    mag = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    mean_spec = mag.mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    power = mean_spec ** 2

    # Harmonic Product Spectrum -> fundamental, used to guard the low edge
    hps = mean_spec.copy()
    for h in range(2, n_harmonics + 1):
        decimated = mean_spec[::h]
        hps[:len(decimated)] *= decimated

    valid = (freqs >= f0_min) & (freqs <= f0_max)
    f0 = freqs[np.argmax(np.where(valid, hps, 0.0))]

    # Band edges from where the spectral energy actually lives.
    # A drone is a dense harmonic comb (dozens of partials into the kHz),
    # so bound by cumulative energy rather than a fixed harmonic count.
    cum = np.cumsum(power) / power.sum()
    low = freqs[np.searchsorted(cum, low_pct)]
    high = freqs[np.searchsorted(cum, high_pct)]

    # Floor the low edge to cut sub-band wind/rumble from the original scene
    # (it overlaps the lowest drone partials but otherwise leaks into the mix).
    low = max(min(low, f0 * 0.8), min_low)
    high = min(high, sr * 0.475)

    return float(low), float(high), float(f0)


def isolate_drone_audio(input_path, output_path, min_low=60.0):

    # Load mono audio
    y, sr = librosa.load(
        input_path,
        sr=None,
        mono=True
    )

    # Normalize
    y = rms_normalize(y)

    # Estimate the drone band from the signal itself. A higher min_low cuts
    # more low-frequency wind/rumble (good for placing the drone in a new
    # scene) at the cost of the drone's lowest harmonics.
    low, high, f0 = estimate_drone_band(y, sr, min_low=min_low)
    print(f"Estimated drone f0={f0:.1f} Hz, band {low:.0f}-{high:.0f} Hz")

    # Remove wind rumble
    y = butter_highpass(
        y,
        sr,
        cutoff=low
    )

    # Focus on drone harmonic range
    y = butter_bandpass(
        y,
        sr,
        low=low,
        high=high
    )

    # Mild adaptive noise reduction
    y = nr.reduce_noise(
        y=y,
        sr=sr,
        stationary=False,
        prop_decrease=0.25
    )

    # Spectral gating, focused on the estimated drone band
    y = spectral_gate(y, sr, low=low, high=high)

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