"""End-to-end framework evaluation.

Runs the full synthetic-generation pipeline on a real drone recording, then
scores the synthetic result against that same recording (the ground truth):

    ground truth (real drone in scene)
        -> isolate drone signature
        -> estimate SNR profile (drone vs background)
        -> mix isolated drone back onto the original background
        -> synthetic clip
        -> compare synthetic vs ground truth (Band Power Ratio)

A small BPR gap means the framework reconstructs a realistic drone-in-scene.

Artifacts are written under --out-dir:
    isolated/<phase>.wav
    profiles/<phase>_snr.json
    synthetic/<phase>.wav
    results/comparison.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for _d in ("drone_isolation", "snr_analysis", "drone_mixer"):
    sys.path.insert(0, str(SRC / _d))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drone_cleaner import isolate_drone_audio  # noqa: E402X
import snr_analysis as snr  # noqa: E402
import mixer  # noqa: E402
from compare import band_power_ratio, mfcc_distance, log_spectral_distance  # noqa: E402


def build_profile(drone_signal, background_signal, sr, smooth=5):
    """SNR profile (smoothed framewise SNR in dB) from drone vs background."""
    bg_power, _, _ = snr.estimate_background_power(background_signal, sr)
    snr_db, _ = snr.compute_snr(drone_signal, sr, bg_power)
    return snr.smooth_curve(snr_db, smooth)


def main():
    parser = argparse.ArgumentParser(
        description="Run the framework on a use case and validate against ground truth."
    )
    parser.add_argument("--ground-truth", required=True, type=Path,
                        help="Real drone-in-scene WAV (also the comparison target).")
    parser.add_argument("--background", required=True, type=Path,
                        help="Background-only WAV from the same scene.")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="Use-case directory to write artifacts into.")
    parser.add_argument("--phase", default="hover")
    parser.add_argument("--snr-offset", type=float, default=0.0,
                        help="dB added to the SNR profile (negative = louder background).")
    parser.add_argument("--smooth", type=int, default=5)
    args = parser.parse_args()

    iso_dir = args.out_dir / "isolated"
    prof_dir = args.out_dir / "profiles"
    syn_dir = args.out_dir / "synthetic"
    res_dir = args.out_dir / "results"
    for d in (iso_dir, prof_dir, syn_dir, res_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Isolate the drone signature from the real recording.
    iso_path = iso_dir / f"{args.phase}.wav"
    isolate_drone_audio(str(args.ground_truth), str(iso_path))

    # 2. SNR profile from the raw drone recording vs the background (common sr).
    drone_raw, sr = snr.load_audio(str(args.ground_truth))
    bg_signal, _ = snr.load_audio(str(args.background), target_sample_rate=sr)
    profile = build_profile(drone_raw, bg_signal, sr, args.smooth) + args.snr_offset
    (prof_dir / f"{args.phase}_snr.json").write_text(
        json.dumps({"phase": args.phase, "smoothed_snr_db": profile.tolist()}, indent=2)
    )

    # 3. Mix isolated drone back onto the ORIGINAL background (validation branch).
    iso_signal, _ = mixer.load_audio(str(iso_path), target_sample_rate=sr)
    bg_for_mix, _ = mixer.load_audio(str(args.background), target_sample_rate=sr)
    synthetic = mixer.dynamic_snr_mix(iso_signal, bg_for_mix, sr, profile)
    syn_path = syn_dir / f"{args.phase}.wav"
    sf.write(str(syn_path), synthetic, sr)

    # 4. Compare synthetic vs ground truth on three metrics.
    #    Each is calibrated against the background (the "totally different"
    #    reference) into a 0-1 similarity: 1 = matches the real recording,
    #    0 = no closer to real than unrelated background noise is.
    gt, bg, syn = str(args.ground_truth), str(args.background), str(syn_path)

    def similarity(dist_syn, dist_far):
        if dist_far <= 1e-9:
            return 1.0
        return float(np.clip(1.0 - dist_syn / dist_far, 0.0, 1.0))

    bpr_syn = band_power_ratio(syn)
    bpr_gt = band_power_ratio(gt)
    bpr_bg = band_power_ratio(bg)
    bpr_sim = similarity(abs(bpr_syn - bpr_gt), abs(bpr_gt - bpr_bg))

    mfcc_dist = mfcc_distance(syn, gt)
    mfcc_sim = similarity(mfcc_dist, mfcc_distance(gt, bg))

    lsd = log_spectral_distance(syn, gt)
    lsd_sim = similarity(lsd, log_spectral_distance(gt, bg))

    result = {
        "use_case": args.out_dir.name,
        "phase": args.phase,
        "snr_offset_db": args.snr_offset,
        "bpr_synthetic": round(bpr_syn, 4),
        "bpr_ground_truth": round(bpr_gt, 4),
        "bpr_gap": round(bpr_syn - bpr_gt, 4),
        "mfcc_distance": round(mfcc_dist, 4),
        "log_spectral_distance_db": round(lsd, 4),
        "similarity": {
            "bpr": round(bpr_sim, 3),
            "mfcc": round(mfcc_sim, 3),
            "log_spectral": round(lsd_sim, 3),
            "overall": round((bpr_sim + mfcc_sim + lsd_sim) / 3.0, 3),
        },
        "synthetic_path": syn,
    }
    (res_dir / "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
