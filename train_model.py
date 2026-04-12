import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("Loading features...")
X = np.load('features/spectrograms.npy')
y = np.load('features/labels.npy')

print(f"Spectrograms shape: {X.shape}")
X_min = X.min()
X_max = X.max()
X = (X - X_min) / (X_max - X_min)
X = X[..., np.newaxis]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

def augment(spec):
    spec = spec.copy()

    # Time shift
    shift = np.random.randint(-30, 30)
    spec = np.roll(spec, shift, axis=1)

    # Multiple frequency masks (SpecAugment style)
    for _ in range(np.random.randint(1, 3)):
        f_start = np.random.randint(0, 100)
        f_width = np.random.randint(5, 25)
        spec[f_start:f_start+f_width, :, :] = 0

    # Multiple time masks
    for _ in range(np.random.randint(1, 3)):
        t_start = np.random.randint(0, 180)
        t_width = np.random.randint(5, 40)
        spec[:, t_start:t_start+t_width, :] = 0

    # Random gain (volume change)
    gain = np.random.uniform(0.7, 1.3)
    spec = spec * gain

    # Gaussian noise
    spec = spec + np.random.normal(0, 0.02, spec.shape)
    spec = np.clip(spec, 0, 1)

    return spec

# Triple augmentation instead of double
X_aug1 = np.array([augment(x.copy()) for x in X_train])
X_aug2 = np.array([augment(x.copy()) for x in X_train])
X_train_full = np.concatenate([X_train, X_aug1, X_aug2], axis=0)
y_train_full = np.concatenate([y_train, y_train, y_train], axis=0)

idx = np.random.permutation(len(X_train_full))
X_train_full = X_train_full[idx]
y_train_full = y_train_full[idx]
print(f"Training set after augmentation: {X_train_full.shape[0]} clips")

model = keras.Sequential([
    keras.layers.Input(shape=X_train.shape[1:]),

    # Block 1
    keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.2),

    # Block 2
    keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.25),

    # Block 3
    keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.3),

    # Block 4
    keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.3),

    keras.layers.GlobalAveragePooling2D(),

    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.4),

    keras.layers.Dense(50, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0008),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

print("\nTraining...")
history = model.fit(
    X_train_full, y_train_full,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=15,
            restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=7, min_lr=0.000001)
    ]
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc*100:.2f}%")
model.save('audio_classifier_model.keras')
print("Model saved!")

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()