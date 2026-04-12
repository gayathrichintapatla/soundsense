import numpy as np
import librosa
import pandas as pd
import os

# Settings — must match training exactly
SAMPLE_RATE = 22050
DURATION = 5
SAMPLES = SAMPLE_RATE * DURATION
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

# ESC-50 class name to label number mapping
CLASS_NAMES = [
    'dog', 'rooster', 'pig', 'cow', 'frog',
    'cat', 'hen', 'insects', 'sheep', 'crow',
    'rain', 'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds',
    'water_drops', 'wind', 'pouring_water', 'toilet_flush', 'thunderstorm',
    'crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing',
    'footsteps', 'laughing', 'brushing_teeth', 'snoring', 'drinking_sipping',
    'door_wood_knock', 'mouse_click', 'keyboard_typing', 'door_wood_creaks', 'can_opening',
    'washing_machine', 'vacuum_cleaner', 'clock_alarm', 'clock_tick', 'glass_breaking',
    'helicopter', 'chainsaw', 'siren', 'car_horn', 'engine',
    'train', 'church_bells', 'airplane', 'fireworks', 'hand_saw'
]

CLASS_TO_LABEL = {name: idx for idx, name in enumerate(CLASS_NAMES)}

def process_audio(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    
    # Pad or trim
    if len(audio) < SAMPLES:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))
    else:
        audio = audio[:SAMPLES]
    
    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT
    )
    
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

# Load existing features
print("Loading existing features...")
X_existing = np.load('features/spectrograms.npy')
y_existing = np.load('features/labels.npy')
print(f"Existing: {X_existing.shape[0]} clips")

# Process your recordings
my_recordings_path = 'my_recordings/'
new_spectrograms = []
new_labels = []
failed = []

print("\nProcessing your recordings...")

for class_folder in os.listdir(my_recordings_path):
    class_path = os.path.join(my_recordings_path, class_folder)
    
    if not os.path.isdir(class_path):
        continue
    
    # Check if class name is valid
    if class_folder not in CLASS_TO_LABEL:
        print(f"  Skipping '{class_folder}' — not a valid ESC-50 class name")
        continue
    
    label = CLASS_TO_LABEL[class_folder]
    files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
    
    print(f"\n  Class: {class_folder} (label {label}) — {len(files)} files")
    
    for file_name in files:
        file_path = os.path.join(class_path, file_name)
        try:
            spec = process_audio(file_path)
            new_spectrograms.append(spec)
            new_labels.append(label)
            print(f"    OK: {file_name}")
        except Exception as e:
            print(f"    Failed: {file_name} — {e}")
            failed.append(file_name)

if len(new_spectrograms) == 0:
    print("\nNo recordings found. Check your my_recordings/ folder structure.")
else:
    # Combine with existing data
    new_spectrograms = np.array(new_spectrograms)
    new_labels = np.array(new_labels)
    
    X_combined = np.concatenate([X_existing, new_spectrograms], axis=0)
    y_combined = np.concatenate([y_existing, new_labels], axis=0)
    
    print(f"\n{'='*40}")
    print(f"Original dataset: {X_existing.shape[0]} clips")
    print(f"Your recordings:  {new_spectrograms.shape[0]} clips")
    print(f"Combined total:   {X_combined.shape[0]} clips")
    print(f"Failed:           {len(failed)} clips")
    
    # Save combined features
    np.save('features/spectrograms.npy', X_combined)
    np.save('features/labels.npy', y_combined)
    
    print(f"\nSaved combined dataset to features/")
    print("Now run: python train_model.py")