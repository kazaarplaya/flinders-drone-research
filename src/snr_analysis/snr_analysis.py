from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


EPSILON = 1e-10
FRAME_LENGTH_MS = 40.0
HOP_LENGTH_MS = 20.0
PHASE_NAMES = [
    "takeoff",
    "hover",
    "altitude_increase",
    "altitude_decrease",
    "landing",
]


def load_audio(file_path: str | Path, target_sample_rate: int | None = None) -> Tuple[np.ndarray, int]:
    """
    Load a WAV file as mono float32 audio.

    - Converts integer PCM to floating point
    - Converts stereo to mono by averaging channels
    - Resamples if a target sample rate is provided
    """
    sample_rate, signal = wavfile.read(str(file_path))
    signal = np.asarray(signal)

    if np.issubdtype(signal.dtype, np.integer):
        signal = signal.astype(np.float32) / np.iinfo(signal.dtype).max
    else:
        signal = signal.astype(np.float32)

    if signal.ndim > 1:
        signal = np.mean(signal, axis=1, dtype=np.float32)

    if target_sample_rate is not None and sample_rate != target_sample_rate:
        gcd = np.gcd(sample_rate, target_sample_rate)
        up = target_sample_rate // gcd
        down = sample_rate // gcd
        signal = resample_poly(signal, up, down).astype(np.float32)
        sample_rate = target_sample_rate

    return np.ravel(signal).astype(np.float32), sample_rate


def frame_signal(signal: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slice a 1D signal into overlapping frames.

    Frame length: 40 ms
    Hop length: 20 ms
    """
    signal = np.ravel(np.asarray(signal, dtype=np.float32))

    frame_length = max(int(round(FRAME_LENGTH_MS * sample_rate / 1000.0)), 1)
    hop_length = max(int(round(HOP_LENGTH_MS * sample_rate / 1000.0)), 1)

    if signal.size < frame_length:
        signal = np.pad(signal, (0, frame_length - signal.size))

    frame_starts = np.arange(0, signal.size - frame_length + 1, hop_length)
    frames = np.stack([signal[start:start + frame_length] for start in frame_starts], axis=0)

    # Use frame centers for the time axis in plots.
    frame_times = (frame_starts + frame_length / 2.0) / sample_rate
    return frames, frame_times


def compute_power(frames: np.ndarray) -> np.ndarray:
    """
    Compute framewise average power:
        P = (1/N) * sum(x[n]^2)
    """
    frames = np.asarray(frames, dtype=np.float32)
    return np.mean(frames ** 2, axis=1)


def estimate_background_power(background_signal: np.ndarray, sample_rate: int) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate background noise power from the background-only clip.

    We use the median frame power to reduce sensitivity to short transients.
    """
    background_frames, background_times = frame_signal(background_signal, sample_rate)
    background_powers = compute_power(background_frames)
    background_power = float(np.median(background_powers))
    return background_power, background_powers, background_times


def compute_snr(
    phase_signal: np.ndarray,
    sample_rate: int,
    background_power: float,
    epsilon: float = EPSILON,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute framewise SNR for one drone phase.

    Model:
        P_s(i) = max(P_x(i) - P_n, epsilon)
        SNR_i = 10 * log10(P_s(i) / P_n)
    """
    phase_frames, frame_times = frame_signal(phase_signal, sample_rate)
    mixture_power = compute_power(phase_frames)

    drone_power = np.maximum(mixture_power - background_power, epsilon)
    safe_background_power = max(background_power, epsilon)

    snr_db = 10.0 * np.log10(drone_power / safe_background_power)
    return snr_db, frame_times


def summarize_snr(snr_db: np.ndarray) -> Dict[str, float]:
    """Return summary statistics for one phase."""
    snr_db = np.asarray(snr_db, dtype=np.float32)
    return {
        "median": float(np.median(snr_db)),
        "mean": float(np.mean(snr_db)),
        "p25": float(np.percentile(snr_db, 25)),
        "p75": float(np.percentile(snr_db, 75)),
        "min": float(np.min(snr_db)),
        "max": float(np.max(snr_db)),
    }


def smooth_curve(values: np.ndarray, window_size: int) -> np.ndarray:
    """Optional moving-average smoothing."""
    values = np.asarray(values, dtype=np.float32)

    if window_size <= 1:
        return values

    kernel = np.ones(window_size, dtype=np.float32) / window_size
    return np.convolve(values, kernel, mode="same")


def plot_phase_snr(
    phase_name: str,
    frame_times: np.ndarray,
    snr_db: np.ndarray,
    *,
    smoothed_snr: np.ndarray | None = None,
) -> None:
    """Plot SNR over time for one phase."""
    plt.figure(figsize=(10, 4))
    plt.plot(frame_times, snr_db, label="Framewise SNR", linewidth=1.5)

    if smoothed_snr is not None:
        plt.plot(frame_times, smoothed_snr, label="Smoothed SNR", linewidth=2.0)

    plt.title(f"{phase_name.replace('_', ' ').title()} SNR")
    plt.xlabel("Time (s)")
    plt.ylabel("SNR (dB)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


def plot_full_recording_snr(
    full_times: np.ndarray,
    full_snr: np.ndarray,
    *,
    smoothed_snr: np.ndarray | None = None,
) -> None:
    """Optional plot across all phases concatenated in time order."""
    plt.figure(figsize=(12, 4))
    plt.plot(full_times, full_snr, label="Framewise SNR", linewidth=1.5)

    if smoothed_snr is not None:
        plt.plot(full_times, smoothed_snr, label="Smoothed SNR", linewidth=2.0)

    plt.title("Full Recording SNR Across All Phases")
    plt.xlabel("Time (s)")
    plt.ylabel("SNR (dB)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


def print_phase_summary(phase_name: str, stats: Dict[str, float]) -> None:
    """Print SNR metrics in the requested format."""
    print(f"{phase_name.replace('_', ' ').title()}:")
    print(f"Median SNR: {stats['median']:.2f} dB")
    print(f"Mean SNR: {stats['mean']:.2f} dB")
    print(f"IQR: [{stats['p25']:.2f}, {stats['p75']:.2f}] dB")
    print(f"Min: {stats['min']:.2f} dB")
    print(f"Max: {stats['max']:.2f} dB")
    print()


def resolve_phase_paths(base_dir: str | Path, phase_names: Iterable[str]) -> Dict[str, Path]:
    """Build expected file paths for each phase WAV file."""
    base_dir = Path(base_dir)
    return {phase: base_dir / f"{phase}.wav" for phase in phase_names}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate framewise and phase-wise SNR for drone audio phases."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Directory containing background.wav and the phase WAV files.",
    )
    parser.add_argument(
        "--background",
        type=Path,
        default=None,
        help="Optional explicit path to background-only WAV.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=0,
        help="Optional moving-average window size in frames.",
    )
    parser.add_argument(
        "--plot-full",
        action="store_true",
        help="Plot an additional SNR curve across all phases.",
    )
    args = parser.parse_args()

    background_path = args.background or (args.base_dir / "background.wav")
    phase_paths = resolve_phase_paths(args.base_dir, PHASE_NAMES)

    background_signal, sample_rate = load_audio(background_path)
    background_power, _, _ = estimate_background_power(background_signal, sample_rate)

    all_times = []
    all_snr = []
    time_offset = 0.0

    for phase_name in PHASE_NAMES:
        phase_signal, phase_sample_rate = load_audio(phase_paths[phase_name], target_sample_rate=sample_rate)

        if phase_sample_rate != sample_rate:
            raise ValueError(f"Sample rate mismatch after loading phase: {phase_name}")

        snr_db, frame_times = compute_snr(phase_signal, sample_rate, background_power)
        stats = summarize_snr(snr_db)
        print_phase_summary(phase_name, stats)

        smoothed_snr = None
        if args.smooth_window > 1:
            smoothed_snr = smooth_curve(snr_db, args.smooth_window)

        plot_phase_snr(phase_name, frame_times, snr_db, smoothed_snr=smoothed_snr)

        if args.plot_full:
            all_times.append(frame_times + time_offset)
            all_snr.append(snr_db)
            if frame_times.size:
                time_offset += float(frame_times[-1] + (HOP_LENGTH_MS / 1000.0))

    if args.plot_full and all_times and all_snr:
        full_times = np.concatenate(all_times)
        full_snr = np.concatenate(all_snr)

        smoothed_full_snr = None
        if args.smooth_window > 1:
            smoothed_full_snr = smooth_curve(full_snr, args.smooth_window)

        plot_full_recording_snr(full_times, full_snr, smoothed_snr=smoothed_full_snr)

    plt.show()


if __name__ == "__main__":
    main()
