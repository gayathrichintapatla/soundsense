from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import tensorflow as tf
import tempfile
import os

app = Flask(__name__)
CORS(app)

# Settings — must match training
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

# Load model once at startup
print("Loading model...")
model = tf.keras.models.load_model('audio_classifier_model.keras')
print("Model ready!")

def preprocess(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    if len(audio) < SAMPLES:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))
    else:
        audio = audio[:SAMPLES]
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE,
        n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - (-80.0)) / (0.0 - (-80.0))
    mel_norm = np.clip(mel_norm, 0, 1)
    return mel_norm[np.newaxis, ..., np.newaxis]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        spec = preprocess(tmp_path)
        preds = model.predict(spec, verbose=0)[0]
        top5 = np.argsort(preds)[::-1][:5]
        results = [
            {'class': CLASS_NAMES[i], 'confidence': round(float(preds[i]) * 100, 2)}
            for i in top5
        ]
        return jsonify({'predictions': results})
    finally:
        os.unlink(tmp_path)

if __name__ == '__main__':
    app.run(debug=False, port=5000)