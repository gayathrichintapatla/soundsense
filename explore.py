import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

df = pd.read_csv('ESC-50/meta/esc50.csv')

print("Total clips:", len(df))
print("Total classes:", df['category'].nunique())
print("\nClips per class:")
print(df['category'].value_counts())

sample_file = 'ESC-50/audio/' + df['filename'].iloc[0]
audio, sample_rate = librosa.load(sample_file, sr=None)

print(f"\nFile: {df['filename'].iloc[0]}")
print(f"Category: {df['category'].iloc[0]}")
print(f"Sample rate: {sample_rate} Hz")
print(f"Duration: {len(audio)/sample_rate:.2f} seconds")

plt.figure(figsize=(12, 4))
librosa.display.waveshow(audio, sr=sample_rate)
plt.title(f"Waveform — {df['category'].iloc[0]}")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("waveform.png")
plt.show()
print("Waveform saved!")