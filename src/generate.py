"""Synthetic drone-audio generator: isolate -> SNR profile -> mix.

SNR profile is measured against the drone's own scene (--background).
--mix-onto picks where the drone lands: omit it to mix back onto the
original scene (validation), or pass a new scene to generate new data.

Writes isolated/, profiles/ and synthetic/ under --out-dir.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SRC = Path(__file__).resolve().parent
for _d in ("drone_isolation", "snr_analysis", "drone_mixer"):
    sys.path.insert(0, str(SRC / _d))

from drone_cleaner import isolate_drone_audio  # noqa: E402
import snr_analysis as snr  # noqa: E402
import mixer  # noqa: E402


def build_profile(drone_signal, background_signal, sr, smooth=5):
    """Smoothed framewise SNR (dB) of the drone against the background."""
    bg_power, _, _ = snr.estimate_background_power(background_signal, sr)
    snr_db, _ = snr.compute_snr(drone_signal, sr, bg_power)
    return snr.smooth_curve(snr_db, smooth)


def generate_synthetic(
    drone_path,
    background_path,
    out_dir,
    mix_onto=None,
    phase="hover",
    name=None,
    snr_offset=0.0,
    smooth=5,
    min_low=60.0,
):
    """Run the pipeline; return the artifact paths.

    drone_path      : real drone-in-scene recording
    background_path : same-scene background (SNR reference)
    mix_onto        : background to mix onto (default: background_path)
    name            : basename for the synthetic WAV (default: phase)
    min_low         : highpass floor (Hz); higher cuts more wind/rumble
    """
    out_dir = Path(out_dir)
    mix_bg_path = Path(mix_onto) if mix_onto else Path(background_path)
    syn_name = name or phase

    iso_dir = out_dir / "isolated"
    prof_dir = out_dir / "profiles"
    syn_dir = out_dir / "synthetic"
    for d in (iso_dir, prof_dir, syn_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Clear previous synthetic outputs so each run leaves only its own.
    for old in syn_dir.glob("*.wav"):
        old.unlink()

    # 1. Isolate the drone.
    iso_path = iso_dir / f"{phase}.wav"
    isolate_drone_audio(str(drone_path), str(iso_path), min_low=min_low)

    # 2. SNR profile: raw drone vs background.
    drone_raw, sr = snr.load_audio(str(drone_path))
    bg_signal, _ = snr.load_audio(str(background_path), target_sample_rate=sr)
    profile = build_profile(drone_raw, bg_signal, sr, smooth) + snr_offset

    prof_path = prof_dir / f"{phase}_snr.json"
    prof_path.write_text(
        json.dumps({"phase": phase, "smoothed_snr_db": profile.tolist()}, indent=2)
    )

    # 3. Mix isolated drone onto the target background.
    iso_signal, _ = mixer.load_audio(str(iso_path), target_sample_rate=sr)
    mix_bg, _ = mixer.load_audio(str(mix_bg_path), target_sample_rate=sr)
    synthetic = mixer.dynamic_snr_mix(iso_signal, mix_bg, sr, profile)

    syn_path = syn_dir / f"{syn_name}.wav"
    sf.write(str(syn_path), synthetic, sr)

    return {
        "isolated": str(iso_path),
        "profile": str(prof_path),
        "synthetic": str(syn_path),
        "sample_rate": sr,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic drone-in-scene clip from a real recording."
    )
    parser.add_argument("--drone", required=True,
                        help="Real drone-in-scene WAV (isolation + SNR source).")
    parser.add_argument("--background", required=True,
                        help="Background-only WAV from the same scene (SNR reference).")
    parser.add_argument("--out-dir", required=True,
                        help="Directory to write artifacts into.")
    parser.add_argument("--mix-onto", default=None,
                        help="Background to mix onto (default: --background).")
    parser.add_argument("--phase", default="hover")
    parser.add_argument("--name", default=None,
                        help="Basename for the synthetic WAV (default: --phase).")
    parser.add_argument("--snr-offset", type=float, default=0.0,
                        help="dB added to the SNR profile (negative = louder background).")
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--min-low", type=float, default=60.0,
                        help="Highpass floor in Hz; raise (e.g. 120-150) to cut "
                             "more wind/rumble when mixing into a new scene.")
    args = parser.parse_args()

    paths = generate_synthetic(
        drone_path=args.drone,
        background_path=args.background,
        out_dir=args.out_dir,
        mix_onto=args.mix_onto,
        phase=args.phase,
        name=args.name,
        snr_offset=args.snr_offset,
        smooth=args.smooth,
        min_low=args.min_low,
    )
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
