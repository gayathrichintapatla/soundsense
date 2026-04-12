import numpy as np
import librosa
import tensorflow as tf
import sys
import os

SAMPLE_RATE = 22050
DURATION = 5
SAMPLES = SAMPLE_RATE * DURATION
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

CLASS_NAMES = [
    'dog','rooster','pig','cow','frog','cat','hen','insects','sheep','crow',
    'rain','sea_waves','crackling_fire','crickets','chirping_birds',
    'water_drops','wind','pouring_water','toilet_flush','thunderstorm',
    'crying_baby','sneezing','clapping','breathing','coughing',
    'footsteps','laughing','brushing_teeth','snoring','drinking_sipping',
    'door_wood_knock','mouse_click','keyboard_typing','door_wood_creaks','can_opening',
    'washing_machine','vacuum_cleaner','clock_alarm','clock_tick','glass_breaking',
    'helicopter','chainsaw','siren','car_horn','engine',
    'train','church_bells','airplane','fireworks','hand_saw'
]

def preprocess_audio(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    if len(audio) < SAMPLES:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))
    else:
        audio = audio[:SAMPLES]
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE,
        n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - (-80.0)) / 80.0
    mel_norm = np.clip(mel_norm, 0, 1)
    return mel_norm[np.newaxis, ..., np.newaxis]

def predict(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found — {file_path}")
        return
    print("Loading model...")
    model = tf.keras.models.load_model('audio_classifier_model.keras')
    spec = preprocess_audio(file_path)
    predictions = model.predict(spec, verbose=0)
    top3 = np.argsort(predictions[0])[::-1][:3]
    print("\n" + "="*40)
    print("PREDICTION RESULTS")
    print("="*40)
    print(f"\nTop prediction:")
    print(f"  {CLASS_NAMES[top3[0]].upper()} ({predictions[0][top3[0]]*100:.1f}% confident)")
    print(f"\nOther possibilities:")
    for i in range(1, 3):
        print(f"  {i+1}. {CLASS_NAMES[top3[i]]} ({predictions[0][top3[i]]*100:.1f}%)")
    print("="*40)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py your_audio.wav")
    else:
        predict(sys.argv[1])