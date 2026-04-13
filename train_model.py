import numpy as np
import tensorflow as tf
from tensorflow import keras
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

# Load features
print("Loading features...")
X = np.load('features/spectrograms.npy')
y = np.load('features/labels.npy')

print(f"Spectrograms shape: {X.shape}")

# Normalize
X_min = X.min()
X_max = X.max()
X = (X - X_min) / (X_max - X_min)
X = X[..., np.newaxis]

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# Augmentation function
def augment(spec):
    spec = spec.copy()
    shift = np.random.randint(-30, 30)
    spec = np.roll(spec, shift, axis=1)
    for _ in range(np.random.randint(1, 3)):
        f_start = np.random.randint(0, 180)
        f_width = np.random.randint(5, 25)
        spec[f_start:f_start+f_width, :, :] = 0
    for _ in range(np.random.randint(1, 3)):
        t_start = np.random.randint(0, 180)
        t_width = np.random.randint(5, 40)
        spec[:, t_start:t_start+t_width, :] = 0
    gain = np.random.uniform(0.7, 1.3)
    spec = spec * gain
    spec = spec + np.random.normal(0, 0.02, spec.shape)
    spec = np.clip(spec, 0, 1)
    return spec

# Mixup function
def mixup(X, y, alpha=0.3):
    indices = np.random.permutation(len(X))
    X2 = X[indices]
    y2 = y[indices]
    lam = np.random.beta(alpha, alpha, len(X))
    lam = np.maximum(lam, 1 - lam)
    X_mixed = np.array([
        lam[i] * X[i] + (1 - lam[i]) * X2[i]
        for i in range(len(X))
    ])
    return X_mixed, y

# Step 1 — Triple augmentation
print("\nAugmenting training data...")
X_aug1 = np.array([augment(x.copy()) for x in X_train])
X_aug2 = np.array([augment(x.copy()) for x in X_train])
X_aug3 = np.array([augment(x.copy()) for x in X_train])
X_train_full = np.concatenate([X_train, X_aug1, X_aug2, X_aug3], axis=0)
y_train_full = np.concatenate([y_train, y_train, y_train, y_train], axis=0)

# Step 2 — Mixup on worst classes
worst_indices = [29, 2, 35, 41, 16, 11, 13, 21, 33, 36]
mask = np.isin(y_train, worst_indices)
X_worst = X_train[mask]
y_worst = y_train[mask]

if len(X_worst) > 0:
    X_mix, y_mix = mixup(X_worst, y_worst)
    X_train_full = np.concatenate([X_train_full, X_mix], axis=0)
    y_train_full = np.concatenate([y_train_full, y_mix], axis=0)
    print(f"Added {len(X_mix)} mixup samples for worst classes")

# Shuffle
idx = np.random.permutation(len(X_train_full))
X_train_full = X_train_full[idx]
y_train_full = y_train_full[idx]

print(f"Total training samples after augmentation: {X_train_full.shape[0]}")

# Class weights — boost worst classes
# Updated class weights based on latest evaluation
class_weight = {i: 1.0 for i in range(50)}

worst_now = [
    'drinking_sipping',   # 0.40
    'pig',                # 0.53
    'washing_machine',    # 0.55
    'chainsaw',           # 0.55
    'wind',               # 0.57
    'sea_waves',          # 0.62
    'crickets',           # 0.67
    'sneezing',           # 0.67
    'door_wood_creaks',   # 0.67
    'vacuum_cleaner',     # 0.67
]

for class_name in worst_now:
    idx_c = CLASS_NAMES.index(class_name)
    class_weight[idx_c] = 2.0

print("Class weights updated for current worst classes")

# Build model
print("\nBuilding model...")
model = keras.Sequential([
    keras.layers.Input(shape=X_train.shape[1:]),

    keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.3),

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

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0008),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
print("\nTraining...")
history = model.fit(
    X_train_full, y_train_full,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=32,
    class_weight=class_weight,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=5,
            min_lr=0.000001
        )
    ]
)

# Evaluate
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc*100:.2f}%")
print(f"Test loss:     {test_loss:.4f}")

model.save('audio_classifier_model.keras')
print("Model saved!")

# Plot
plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()
print("Training history saved!")