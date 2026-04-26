import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--audio",
    type=str,
    required=True,
    help="Path to the audio file"
)

args = parser.parse_args()

signal, sample_rate = librosa.load(args.audio)

mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=20)

plt.figure(figsize=(10, 4))

# Display the MFCCs
librosa.display.specshow(mfccs, x_axis='time', sr=sample_rate)

plt.colorbar(format='%+2.0f dB')
plt.title('MFCC')
plt.xlabel('Time')
plt.ylabel('MFCC Coefficients')
plt.tight_layout()
plt.show()
