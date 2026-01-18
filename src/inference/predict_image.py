import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
import numpy as np
from PIL import Image

# -------------------------------
# 1. Paths
# -------------------------------
MODEL_PATH = "models/vehicle10_mobilenet_best.keras"
CLASS_NAMES = [
    "bicycle", "boat", "bus", "car", "helicopter",
    "minibus", "motorcycle", "taxi", "train", "truck"
]

IMAGE_SIZE = (224, 224)
PREDICTION_DIR = "predictions"
os.makedirs(PREDICTION_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(PREDICTION_DIR, "predictions.txt")

# -------------------------------
# 2. Load model
# -------------------------------
print("[INFO] Loading trained model...")
model = keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded successfully.")

# -------------------------------
# 3. Image preprocessing
# -------------------------------
def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------------------
# 4. Predict function
# -------------------------------
def predict_image(image_path):
    image = load_and_preprocess_image(image_path)
    predictions = model.predict(image)
    confidence = np.max(predictions)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    return predicted_class, confidence

# -------------------------------
# 5. Run prediction
# -------------------------------
if __name__ == "__main__":

    image_path = input("Enter image path: ").strip()

    if not os.path.exists(image_path):
        print("[ERROR] Image file not found.")
        exit()

    predicted_class, confidence = predict_image(image_path)

    result = {
        "image": image_path,
        "prediction": predicted_class,
        "confidence": float(confidence),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Print to terminal
    print("\n=== Prediction Result ===")
    print(f"Image      : {image_path}")
    print(f"Prediction : {predicted_class}")
    print(f"Confidence : {confidence:.4f}")

    # Save to file
    with open(OUTPUT_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")

    print(f"\n[SAVED] Prediction written to {OUTPUT_FILE}")
