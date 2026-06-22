from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


EPSILON = 1e-10

FRAME_LENGTH_MS = 40.0
HOP_LENGTH_MS = 20.0


def load_audio(file_path: str | Path, target_sample_rate: int | None = None):

    sample_rate, signal = wavfile.read(str(file_path))
    signal = np.asarray(signal)

    # Convert PCM -> float
    if np.issubdtype(signal.dtype, np.integer):
        signal = signal.astype(np.float32) / np.iinfo(signal.dtype).max
    else:
        signal = signal.astype(np.float32)

    # Stereo -> mono
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

    # Resample if needed
    if target_sample_rate is not None and sample_rate != target_sample_rate:

        gcd = np.gcd(sample_rate, target_sample_rate)

        up = target_sample_rate // gcd
        down = sample_rate // gcd

        signal = resample_poly(signal, up, down).astype(np.float32)

        sample_rate = target_sample_rate

    return signal.astype(np.float32), sample_rate


def save_audio(output_path: str | Path, signal: np.ndarray, sample_rate: int):

    signal = np.clip(signal, -1.0, 1.0)

    signal_int16 = (signal * 32767).astype(np.int16)

    wavfile.write(str(output_path), sample_rate, signal_int16)


def frame_signal(signal: np.ndarray, sample_rate: int):

    frame_length = max(
        int(round(FRAME_LENGTH_MS * sample_rate / 1000.0)),
        1
    )

    hop_length = max(
        int(round(HOP_LENGTH_MS * sample_rate / 1000.0)),
        1
    )

    if len(signal) < frame_length:
        signal = np.pad(signal, (0, frame_length - len(signal)))

    starts = np.arange(
        0,
        len(signal) - frame_length + 1,
        hop_length
    )

    frames = np.stack([
        signal[start:start + frame_length]
        for start in starts
    ])

    return frames


def match_length(signal: np.ndarray, target_length: int):

    if len(signal) >= target_length:
        return signal[:target_length]

    repeats = int(np.ceil(target_length / len(signal)))

    tiled = np.tile(signal, repeats)

    return tiled[:target_length]


def overlap_add(frames: np.ndarray, sample_rate: int):

    hop_length = max(
        int(round(HOP_LENGTH_MS * sample_rate / 1000.0)),
        1
    )

    num_frames, frame_length = frames.shape

    output_length = (
        hop_length * (num_frames - 1)
        + frame_length
    )

    output = np.zeros(output_length, dtype=np.float32)

    for i, frame in enumerate(frames):

        start = i * hop_length
        end = start + frame_length

        output[start:end] += frame

    return output


def load_snr_profile(json_path: str | Path):

    with open(json_path, "r") as f:
        data = json.load(f)

    return np.array(
        data["smoothed_snr_db"],
        dtype=np.float32
    )


def dynamic_snr_mix(
    drone_signal: np.ndarray,
    background_signal: np.ndarray,
    sample_rate: int,
    snr_profile: np.ndarray,
    min_snr_db: float = -10.0,
):

    # Match lengths
    background_signal = match_length(
        background_signal,
        len(drone_signal)
    )

    # Frame signals
    drone_frames = frame_signal(
        drone_signal,
        sample_rate
    )

    background_frames = frame_signal(
        background_signal,
        sample_rate
    )

    # Match frame counts
    num_frames = min(
        len(drone_frames),
        len(background_frames),
        len(snr_profile)
    )

    drone_frames = drone_frames[:num_frames]
    background_frames = background_frames[:num_frames]
    snr_profile = snr_profile[:num_frames]

    mixed_frames = []

    for drone_frame, noise_frame, target_snr_db in zip(
        drone_frames,
        background_frames,
        snr_profile
    ):

        # Clamp the target so silent-drone frames (whose profile SNR can floor
        # at huge negative values) don't blast the background
        target_snr_db = max(float(target_snr_db), min_snr_db)

        drone_power = np.mean(drone_frame ** 2)

        noise_power = np.mean(noise_frame ** 2)

        noise_power = max(noise_power, EPSILON)

        target_linear = 10 ** (
            target_snr_db / 10.0
        )

        noise_scale = np.sqrt(
            drone_power /
            (noise_power * target_linear)
        )

        scaled_noise = noise_frame * noise_scale

        mixed_frame = (
            drone_frame
            + scaled_noise
        )

        mixed_frames.append(mixed_frame)

    mixed_frames = np.stack(mixed_frames)

    # Reconstruct
    output_signal = overlap_add(
        mixed_frames,
        sample_rate
    )

    # Prevent clipping
    peak = np.max(np.abs(output_signal))

    if peak > 1.0:
        output_signal = output_signal / peak

    return output_signal.astype(np.float32)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Mix an isolated drone onto a background at a target SNR profile."
    )
    parser.add_argument("drone", help="Isolated drone WAV.")
    parser.add_argument("background", help="Background WAV to mix onto.")
    parser.add_argument("snr_profile", help="SNR profile JSON (smoothed_snr_db).")
    parser.add_argument("output", help="Output synthetic WAV path.")
    args = parser.parse_args()

    drone_signal, sample_rate = load_audio(args.drone)
    background_signal, _ = load_audio(args.background, target_sample_rate=sample_rate)
    snr_profile = load_snr_profile(args.snr_profile)

    mixed_signal = dynamic_snr_mix(
        drone_signal,
        background_signal,
        sample_rate,
        snr_profile,
    )

    save_audio(args.output, mixed_signal, sample_rate)
    print(f"Synthetic audio generated → {args.output}")


if __name__ == "__main__":
    main()