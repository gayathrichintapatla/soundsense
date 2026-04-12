import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

# Load raw audio (not spectrograms — YAMNet handles its own features)
print("Loading audio data...")
audio_data = np.load('processed_data/audio.npy')  # shape (2000, 110250)
labels = np.load('processed_data/labels.npy')

# Split
X_train, X_temp, y_train, y_temp = train_test_split(
    audio_data, labels, test_size=0.3, random_state=42, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Load YAMNet from TensorFlow Hub
print("\nLoading YAMNet from TensorFlow Hub...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Extract YAMNet embeddings for all clips
def get_embeddings(audio_array):
    embeddings_list = []
    for i, audio in enumerate(audio_array):
        audio_tensor = tf.cast(audio, tf.float32)
        _, embeddings, _ = yamnet_model(audio_tensor)
        # Average embeddings across time
        embedding = tf.reduce_mean(embeddings, axis=0)
        embeddings_list.append(embedding.numpy())
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(audio_array)}")
    return np.array(embeddings_list)

print("\nExtracting YAMNet embeddings for training set...")
X_train_emb = get_embeddings(X_train)
print("Extracting YAMNet embeddings for validation set...")
X_val_emb = get_embeddings(X_val)
print("Extracting YAMNet embeddings for test set...")
X_test_emb = get_embeddings(X_test)

print(f"\nEmbedding shape: {X_train_emb.shape}")

# Build a simple classifier on top of YAMNet embeddings
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024,)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(50, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("\nTraining classifier on YAMNet embeddings...")
history = model.fit(
    X_train_emb, y_train,
    validation_data=(X_val_emb, y_val),
    epochs=80,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=0.00001
        )
    ]
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test_emb, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc*100:.2f}%")
print(f"Test loss:     {test_loss:.4f}")

model.save('yamnet_classifier_model.keras')
print("Model saved as yamnet_classifier_model.keras")

# Plot
plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('YAMNet Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('YAMNet Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('yamnet_training_history.png')
plt.show()