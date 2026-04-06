from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio(file_path: str | Path) -> Tuple[np.ndarray, int]:
    """Load an audio file and return the waveform and sample rate."""
    signal, sample_rate = sf.read(str(file_path), dtype="float32", always_2d=False)
    return signal, sample_rate


def preprocess_audio(
    signal: np.ndarray,
    sample_rate: int,
    target_sample_rate: int | None = None,
) -> Tuple[np.ndarray, int]:
    """
    Prepare an audio signal for mixing.

    Steps:
    1. Convert multi-channel audio to mono by averaging channels.
    2. Resample to the target sample rate when needed.
    3. Ensure the result is a 1D float32 NumPy array.
    """
    signal = np.asarray(signal, dtype=np.float32)

    # Convert stereo or multi-channel audio to mono so both sources share
    # the same representation before we attempt mixing.
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1, dtype=np.float32)

    processed_sample_rate = sample_rate

    # Resample only when a target rate is provided and differs from the input.
    if target_sample_rate is not None and sample_rate != target_sample_rate:
        gcd = np.gcd(sample_rate, target_sample_rate)
        up = target_sample_rate // gcd
        down = sample_rate // gcd
        signal = resample_poly(signal, up, down).astype(np.float32)
        processed_sample_rate = target_sample_rate

    return np.ravel(signal).astype(np.float32), processed_sample_rate


def match_length(
    signal1: np.ndarray,
    signal2: np.ndarray,
    *,
    loop_shorter: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two 1D signals to the same length.

    The first signal acts as the reference length. This matches a common
    foreground/background workflow where the drone clip defines the desired
    output duration and the background is looped or truncated to fit.
    """
    signal1 = np.ravel(np.asarray(signal1, dtype=np.float32))
    signal2 = np.ravel(np.asarray(signal2, dtype=np.float32))

    target_length = len(signal1)

    if len(signal2) == target_length:
        return signal1, signal2

    if len(signal2) > target_length:
        return signal1, signal2[:target_length]

    if len(signal2) == 0:
        raise ValueError("signal2 is empty and cannot be matched in length.")

    if loop_shorter:
        repeats = int(np.ceil(target_length / len(signal2)))
        signal2 = np.tile(signal2, repeats)[:target_length]
    else:
        signal2 = np.pad(signal2, (0, target_length - len(signal2)))

    return signal1, signal2.astype(np.float32)


def mix_audio(
    drone: np.ndarray,
    background: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    Mix two aligned signals with a weighted sum.

    Baseline formula:
        y(t) = alpha * drone(t) + beta * background(t)
    """
    drone = np.asarray(drone, dtype=np.float32)
    background = np.asarray(background, dtype=np.float32)

    if drone.shape != background.shape:
        raise ValueError("drone and background must have the same shape before mixing.")

    return alpha * drone + beta * background


def normalize_audio(signal: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """
    Peak-normalize a signal to reduce clipping risk when saving to file.

    If the signal is silent, it is returned unchanged.
    """
    signal = np.asarray(signal, dtype=np.float32)
    max_abs = np.max(np.abs(signal)) if signal.size else 0.0

    if max_abs == 0.0:
        return signal

    return (signal / max_abs) * peak


def save_audio(output_path: str | Path, signal: np.ndarray, sample_rate: int) -> None:
    """Save a mono waveform to disk, creating parent folders as needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), np.asarray(signal, dtype=np.float32), sample_rate)


def create_mixture(
    drone_path: str | Path,
    background_path: str | Path,
    output_path: str | Path,
    *,
    alpha: float = 0.7,
    beta: float = 0.3,
    target_sample_rate: int | None = None,
) -> Tuple[np.ndarray, int]:
    """
    End-to-end helper for creating and saving a baseline mixture.

    This wrapper keeps the pipeline modular while giving downstream scripts
    a single entry point they can replace with a more advanced mixer later.
    """
    drone_signal, drone_sample_rate = load_audio(drone_path)
    background_signal, background_sample_rate = load_audio(background_path)

    # Use the drone clip sample rate as the default reference so the
    # foreground signal preserves its original timing unless overridden.
    reference_sample_rate = target_sample_rate or drone_sample_rate

    drone_signal, sample_rate = preprocess_audio(
        drone_signal,
        drone_sample_rate,
        target_sample_rate=reference_sample_rate,
    )
    background_signal, _ = preprocess_audio(
        background_signal,
        background_sample_rate,
        target_sample_rate=reference_sample_rate,
    )

    drone_signal, background_signal = match_length(drone_signal, background_signal)
    mixed_signal = mix_audio(drone_signal, background_signal, alpha=alpha, beta=beta)
    normalized_signal = normalize_audio(mixed_signal)
    save_audio(output_path, normalized_signal, sample_rate)

    return normalized_signal, sample_rate


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    drone_file = project_root / "data" / "raw" / "drone" / "DJI_Matrice_200_38.wav"
    background_file = project_root / "data" / "raw" / "background" / "empty_warehouse.flac"
    output_file = project_root / "data" / "processed" / "mixes" / "drone_warehouse_mix.wav"

    # Baseline example:
    # - load the clean drone clip and warehouse ambience
    # - preprocess to mono and a shared sample rate
    # - match durations by looping/truncating the background
    # - mix with fixed weights
    # - normalize and save the result
    mixed_signal, sample_rate = create_mixture(
        drone_file,
        background_file,
        output_file,
        alpha=0.7,
        beta=0.3,
    )

    print(f"Saved mixed audio to: {output_file}")
    print(f"Output sample rate: {sample_rate} Hz")
    print(f"Number of samples: {len(mixed_signal)}")
