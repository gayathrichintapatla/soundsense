import numpy as np
import librosa
import os

SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
N_MFCC = 40

print("Loading audio...")
audio_data = np.load('processed_data/audio.npy')
labels = np.load('processed_data/labels.npy')

print(f"Extracting combined features for {len(audio_data)} clips...")

all_features = []

for i, audio in enumerate(audio_data):
    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE,
        n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE,
        n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT
    )

    # Delta MFCC (rate of change of MFCC)
    mfcc_delta = librosa.feature.delta(mfcc)

    # Stack all features vertically
    combined = np.vstack([mel_db, mfcc, mfcc_delta])
    all_features.append(combined)

    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(audio_data)} done...")

all_features = np.array(all_features)
print(f"\nCombined feature shape: {all_features.shape}")

os.makedirs('features', exist_ok=True)
np.save('features/combined_features.npy', all_features)
np.save('features/labels.npy', labels)
print("Saved to features/combined_features.npy")