
from pathlib import Path
import json

import numpy as np
from PIL import Image
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "tomato_disease_model.keras"
METADATA_PATH = PROJECT_ROOT / "backend" / "models" / "model_metadata.json"

IMAGE_SIZE = (160, 160)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

if not METADATA_PATH.exists():
    raise FileNotFoundError(f"Model metadata not found: {METADATA_PATH}")

model = keras.models.load_model(MODEL_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

CLASS_NAMES = metadata["class_names"]

if len(CLASS_NAMES) != model.output_shape[-1]:
    raise ValueError("Model classes and metadata do not match.")


def predict_image(image: Image.Image) -> dict:
    image = image.convert("RGB").resize(IMAGE_SIZE)

    image_array = keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index]) * 100

    top_indices = np.argsort(probabilities)[::-1][:3]

    top_predictions = [
        {
            "class_name": CLASS_NAMES[int(index)],
            "confidence_percent": round(
                float(probabilities[index]) * 100, 2
            ),
        }
        for index in top_indices
    ]

    return {
        "predicted_class": predicted_class,
        "confidence_percent": round(confidence, 2),
        "top_predictions": top_predictions,
    }
