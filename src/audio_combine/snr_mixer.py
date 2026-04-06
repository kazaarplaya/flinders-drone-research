# -*- coding: utf-8 -*-

# Import required libraries
import argparse      # For reading command-line arguments
import array         # For writing raw audio data
import math
import numpy as np   # For numerical operations
import random        # For selecting random noise segment
import wave          # For reading/writing .wav files


# -------------------------------
# Parse command-line arguments
# -------------------------------
def get_args():
    parser = argparse.ArgumentParser()

    # Path to clean (signal) audio file
    parser.add_argument('--clean_file', type=str, required=True)

    # Path to noise audio file
    parser.add_argument('--noise_file', type=str, required=True)

    # Output file for the mixed audio
    parser.add_argument('--output_mixed_file', type=str, default='', required=True)

    # Optional outputs (not used later in code)
    parser.add_argument('--output_clean_file', type=str, default='')
    parser.add_argument('--output_noise_file', type=str, default='')

    # Desired Signal-to-Noise Ratio (in dB)
    parser.add_argument('--snr', type=float, default='', required=True)

    args = parser.parse_args()
    return args


# -------------------------------
# Calculate target noise RMS
# based on desired SNR
# -------------------------------
def cal_adjusted_rms(clean_rms, snr):
    # SNR formula:
    # SNR = 20 * log10(clean_rms / noise_rms)
    # Rearranged:
    # noise_rms = clean_rms / (10^(snr/20))

    a = float(snr) / 20
    noise_rms = clean_rms / (10**a)
    return noise_rms


# -------------------------------
# Convert wave file to amplitude array
# -------------------------------
def cal_amp(wf):
    # Read all audio frames (raw bytes)
    buffer = wf.readframes(wf.getnframes())

    # Convert raw bytes into NumPy array of int16
    # int16 is standard for 16-bit audio
    amplitude = np.frombuffer(buffer, dtype="int16").astype(np.float64)

    return amplitude


# -------------------------------
# Compute RMS (signal strength)
# -------------------------------
def cal_rms(amp):
    # RMS = sqrt(mean(x^2))
    return np.sqrt(np.mean(np.square(amp), axis=-1))


# -------------------------------
# Save waveform to file
# -------------------------------
def save_waveform(output_path, params, amp):
    # Create output wave file
    output_file = wave.Wave_write(output_path)

    # Copy original audio parameters:
    # (channels, sample width, frame rate, etc.)
    output_file.setparams(params)

    # Convert float64 back to int16 and write to file
    output_file.writeframes(
        array.array('h', amp.astype(np.int16)).tobytes()
    )

    output_file.close()


# -------------------------------
# Main program
# -------------------------------
if __name__ == '__main__':
    # Get user inputs
    args = get_args()

    clean_file = args.clean_file
    noise_file = args.noise_file

    # Open audio files
    clean_wav = wave.open(clean_file, "r")
    noise_wav = wave.open(noise_file, "r")

    # Convert audio to amplitude arrays
    clean_amp = cal_amp(clean_wav)
    noise_amp = cal_amp(noise_wav)

    # Compute RMS of clean signal
    clean_rms = cal_rms(clean_amp)

    # -------------------------------
    # Match noise length to clean signal
    # -------------------------------

    # Random starting point in noise file
    start = random.randint(0, len(noise_amp) - len(clean_amp))

    # Extract a segment of noise with same length as clean signal
    divided_noise_amp = noise_amp[start: start + len(clean_amp)]

    # Compute RMS of selected noise segment
    noise_rms = cal_rms(divided_noise_amp)

    # -------------------------------
    # Adjust noise level to match SNR
    # -------------------------------

    snr = args.snr

    # Calculate desired noise RMS based on SNR
    adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)

    # Scale noise so its RMS matches the target
    adjusted_noise_amp = divided_noise_amp * (adjusted_noise_rms / noise_rms)

    # -------------------------------
    # Mix clean signal + noise
    # -------------------------------
    mixed_amp = clean_amp + adjusted_noise_amp

    # -------------------------------
    # Prevent clipping (overflow)
    # -------------------------------

    # int16 range limits
    max_int16 = np.iinfo(np.int16).max   # 32767
    min_int16 = np.iinfo(np.int16).min   # -32768

    # If values exceed allowed range, scale everything down
    if mixed_amp.max(axis=0) > max_int16 or mixed_amp.min(axis=0) < min_int16:

        # Determine scaling factor based on largest overflow
        if mixed_amp.max(axis=0) >= abs(mixed_amp.min(axis=0)):
            reduction_rate = max_int16 / mixed_amp.max(axis=0)
        else:
            reduction_rate = min_int16 / mixed_amp.min(axis=0)

        # Apply scaling
        mixed_amp = mixed_amp * reduction_rate
        clean_amp = clean_amp * reduction_rate

    # -------------------------------
    # Save final mixed audio
    # -------------------------------
    save_waveform(args.output_mixed_file, clean_wav.getparams(), mixed_amp)