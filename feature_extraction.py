import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import os

SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
PROCESSED_PATH = 'processed_data/'
SAVE_PATH = 'features/'
os.makedirs(SAVE_PATH, exist_ok=True)

print("Loading preprocessed audio...")
audio_data = np.load(PROCESSED_PATH + 'audio.npy')
labels = np.load(PROCESSED_PATH + 'labels.npy')

print(f"Audio shape: {audio_data.shape}")
print("Extracting Mel spectrograms...")
all_spectrograms = []

for i, audio in enumerate(audio_data):
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE,
        n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    all_spectrograms.append(mel_spec_db)
    if (i + 1) % 200 == 0:
        print(f"  Extracted {i+1}/{len(audio_data)}...")

all_spectrograms = np.array(all_spectrograms)
np.save(SAVE_PATH + 'spectrograms.npy', all_spectrograms)
np.save(SAVE_PATH + 'labels.npy', labels)

print(f"Spectrogram shape: {all_spectrograms.shape}")
print(f"Saved to {SAVE_PATH}")

df = pd.read_csv(PROCESSED_PATH + 'metadata.csv')
plt.figure(figsize=(12, 5))
librosa.display.specshow(
    all_spectrograms[0], sr=SAMPLE_RATE,
    hop_length=HOP_LENGTH, x_axis='time', y_axis='mel'
)
plt.colorbar(format='%+2.0f dB')
plt.title(f"Mel Spectrogram — {df['category'].iloc[0]}")
plt.tight_layout()
plt.savefig('spectrogram_sample.png')
plt.show()