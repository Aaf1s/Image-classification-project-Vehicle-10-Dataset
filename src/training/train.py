import os
import shutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -------------------------------
# 1. Environment setup (CPU-safe)
# -------------------------------
tf.config.threading.set_intra_op_parallelism_threads(6)
tf.config.threading.set_inter_op_parallelism_threads(6)

# -------------------------------
# 2. Paths and cache cleanup
# -------------------------------
dataset_path = "Dataset/vehicle-10"
os.makedirs("models", exist_ok=True)

if os.path.exists(".tf_cache"):
    print("[INFO] Clearing old cache...")
    shutil.rmtree(".tf_cache")
os.makedirs(".tf_cache", exist_ok=True)

# -------------------------------
# 3. Dataset configuration
# -------------------------------
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 16
VAL_SPLIT = 0.2
SHUFFLE_BUFFER = 256
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

print(f"[INFO] Loading dataset from: {dataset_path}")

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=VAL_SPLIT,
    subset="training",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"[INFO] Classes: {class_names} (n={num_classes})")

# -------------------------------
# 4. Data pipeline & preprocessing
# -------------------------------
normalization = layers.Rescaling(1.0 / 255.0)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

def preprocess(x, y):
    x = normalization(x)
    return x, y

train_ds = (
    train_ds
    .shuffle(SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True)
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .cache(".tf_cache/train.cache")
    .prefetch(AUTOTUNE)
)

val_ds = (
    val_ds
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .cache(".tf_cache/val.cache")
    .prefetch(AUTOTUNE)
)

# -------------------------------
# 5. Base model (MobileNetV2)
# -------------------------------
base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,       # exclude original ImageNet classifier
    weights="imagenet"
)

base_model.trainable = False  # freeze all layers for feature extraction

# -------------------------------
# 6. Build full model
# -------------------------------
model = keras.Sequential([
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary(print_fn=lambda s: print("[MODEL]", s))

# -------------------------------
# 7. Callbacks
# -------------------------------
best_ckpt_path = "models/vehicle10_mobilenet_best.keras"
last_ckpt_path = "models/vehicle10_mobilenet_last.keras"

callbacks = [
    keras.callbacks.ModelCheckpoint(
        best_ckpt_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# -------------------------------
# 8. Phase 1 — Feature Extraction
# -------------------------------
EPOCHS = 10
print(f"[INFO] Phase 1: Feature Extraction ({EPOCHS} epochs, base frozen)")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# -------------------------------
# 9. Phase 2 — Fine-Tuning
# -------------------------------
print("\n[INFO] Phase 2: Fine-Tuning top layers...")

# Unfreeze the top 30% of layers in the base model
fine_tune_at = int(len(base_model.layers) * 0.7)
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # lower LR for fine-tuning
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

EPOCHS_FINE = 10
print(f"[INFO] Fine-Tuning for {EPOCHS_FINE} epochs (top {100 - 70}% of layers unfrozen)")

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    callbacks=callbacks,
    verbose=1
)

# -------------------------------
# 10. Save final model
# -------------------------------
model.save(last_ckpt_path)
print(f"[INFO] ✅ Best model saved to: {best_ckpt_path}")
print(f"[INFO] ✅ Final snapshot saved to: {last_ckpt_path}")

# -------------------------------
# 11. Save training history for evaluation
# -------------------------------
import json

history_path = "models/training_history.json"
print(f"[INFO] Saving training history to: {history_path}")

with open(history_path, "w") as f:
    json.dump(history.history, f)

print("[INFO] Training history saved successfully.")
