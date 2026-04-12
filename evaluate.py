import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

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
print("Loading data...")
X = np.load('features/spectrograms.npy')
y = np.load('features/labels.npy')

X_min, X_max = X.min(), X.max()
X = (X - X_min) / (X_max - X_min)
X = X[..., np.newaxis]

_, X_temp, _, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
_, X_test, _, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Load model
print("Loading model...")
model = tf.keras.models.load_model('audio_classifier_model.keras')

# Predict
print("Evaluating...")
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Per class accuracy
print("\n" + "="*60)
print("PER CLASS ACCURACY")
print("="*60)
report = classification_report(y_test, y_pred,
    target_names=CLASS_NAMES, output_dict=True)

# Sort classes by accuracy (worst first)
class_scores = []
for name in CLASS_NAMES:
    if name in report:
        class_scores.append((name, report[name]['f1-score']))

class_scores.sort(key=lambda x: x[1])

print("\nWorst performing classes:")
for name, score in class_scores[:10]:
    print(f"  {name:<25} F1: {score:.2f}")

print("\nBest performing classes:")
for name, score in class_scores[-10:]:
    print(f"  {name:<25} F1: {score:.2f}")

# Confusion matrix — only show worst 15 classes
worst_classes = [name for name, _ in class_scores[:15]]
worst_indices = [CLASS_NAMES.index(c) for c in worst_classes]

mask = np.isin(y_test, worst_indices)
y_test_sub = y_test[mask]
y_pred_sub = y_pred[mask]

cm = confusion_matrix(y_test_sub, y_pred_sub, labels=worst_indices)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d',
    xticklabels=worst_classes,
    yticklabels=worst_classes,
    cmap='Blues')
plt.title('Confusion Matrix — 15 Worst Classes')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("\nConfusion matrix saved as confusion_matrix.png")
print("Overall accuracy:", report['accuracy']*100, "%")