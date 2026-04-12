import pandas as pd
import numpy as np
import librosa
import os

SAMPLE_RATE = 22050
DURATION = 5
SAMPLES = SAMPLE_RATE * DURATION
CSV_PATH = 'ESC-50/meta/esc50.csv'
AUDIO_PATH = 'ESC-50/audio/'
SAVE_PATH = 'processed_data/'

os.makedirs(SAVE_PATH, exist_ok=True)
df = pd.read_csv(CSV_PATH)

print(f"Processing {len(df)} audio files...")
all_audio = []
all_labels = []

for i, row in df.iterrows():
    file_path = AUDIO_PATH + row['filename']
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        if len(audio) < SAMPLES:
            audio = np.pad(audio, (0, SAMPLES - len(audio)))
        else:
            audio = audio[:SAMPLES]
        all_audio.append(audio)
        all_labels.append(row['target'])
        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(df)} files...")
    except Exception as e:
        print(f"  Failed: {row['filename']} — {e}")

all_audio = np.array(all_audio)
all_labels = np.array(all_labels)

np.save(SAVE_PATH + 'audio.npy', all_audio)
np.save(SAVE_PATH + 'labels.npy', all_labels)
df.to_csv(SAVE_PATH + 'metadata.csv', index=False)

print(f"Done! Shape: {all_audio.shape}")
print(f"Saved to {SAVE_PATH}")