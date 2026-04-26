import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--audio",
    type=str,
    required=True,
    help="Path to the audio file"
)

args = parser.parse_args()

# Load audio
signal, sample_rate = librosa.load(args.audio, sr=None)

# Compute STFT
stft = librosa.stft(signal)

# Convert amplitude to decibels
spectrogram_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

# Plot spectrogram
plt.figure(figsize=(10, 4))

librosa.display.specshow(
    spectrogram_db,
    sr=sample_rate,
    x_axis="time",
    y_axis="hz"
)

plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram")
plt.tight_layout()
plt.show()