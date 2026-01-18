import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

# -------------------------------
# 1. Paths and constants
# -------------------------------
MODEL_PATH = "models/vehicle10_mobilenet_best.keras"
DATASET_PATH = "Dataset/vehicle-10"
HISTORY_PATH = "models/training_history.json"  # optional
REPORTS_DIR = "reports"

os.makedirs(REPORTS_DIR, exist_ok=True)

IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 16
VAL_SPLIT = 0.2
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

# -------------------------------
# 2. Load model
# -------------------------------
print("[INFO] Loading trained model...")
model = keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded successfully.\n")

# -------------------------------
# 3. Load validation dataset
# -------------------------------
print("[INFO] Loading validation dataset...")
val_raw = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = val_raw.class_names
num_classes = len(class_names)
print(f"[INFO] Classes: {class_names} (n={num_classes})")

# Normalization to match training pipeline
normalization = keras.layers.Rescaling(1.0 / 255.0)

val_ds = (
    val_raw
    .map(lambda x, y: (normalization(x), y), num_parallel_calls=AUTOTUNE)
    .cache()
    .prefetch(AUTOTUNE)
)

# -------------------------------
# 4. Evaluate model
# -------------------------------
print("[INFO] Evaluating model on validation set...")
loss, acc = model.evaluate(val_ds)
print(f"\n✅ Validation Accuracy: {acc:.4f}")
print(f"✅ Validation Loss: {loss:.4f}\n")

# -------------------------------
# 5. Generate predictions
# -------------------------------
print("[INFO] Generating predictions...")
y_true = np.concatenate([y for _, y in val_ds], axis=0)
y_pred_probs = model.predict(val_ds, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# -------------------------------
# 6. Classification Report
# -------------------------------
print("[INFO] Creating classification report...")
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("\n" + report)

# Save to file
report_path = os.path.join(REPORTS_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)
print(f"[SAVED] Classification report -> {report_path}")

# -------------------------------
# 7. Confusion Matrix
# -------------------------------
print("[INFO] Generating confusion matrix...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - MobileNetV2 on Vehicle-10")
cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
plt.savefig(cm_path, bbox_inches="tight")
plt.close()
print(f"[SAVED] Confusion matrix image -> {cm_path}")

# -------------------------------
# 8. Accuracy & Loss Curves
# -------------------------------
print("[INFO] Checking for training history...")

if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, "r") as f:
        history = json.load(f)

    print("[INFO] Plotting training accuracy and loss curves...")
    epochs = range(1, len(history['accuracy']) + 1)

    # Accuracy plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["accuracy"], label="Training Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    acc_path = os.path.join(REPORTS_DIR, "accuracy_curve.png")
    plt.savefig(acc_path, bbox_inches="tight")
    plt.close()

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["loss"], label="Training Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    loss_path = os.path.join(REPORTS_DIR, "loss_curve.png")
    plt.savefig(loss_path, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] Accuracy curve -> {acc_path}")
    print(f"[SAVED] Loss curve -> {loss_path}")

else:
    print("[WARN] No training history found. Skipping accuracy/loss plots.")
    print("Tip: You can modify your training script to save history as JSON.")

# -------------------------------
# 9. Preview Predictions
# -------------------------------
print("\n[INFO] Previewing a few predictions:")
for images, labels in val_ds.take(1):
    preds = model.predict(images)
    pred_classes = np.argmax(preds, axis=1)
    for i in range(5):
        true_label = class_names[labels[i].numpy()]
        pred_label = class_names[pred_classes[i]]
        print(f"Image {i+1}: True → {true_label:10s} | Predicted → {pred_label}")

print("\n✅ Evaluation complete!")
