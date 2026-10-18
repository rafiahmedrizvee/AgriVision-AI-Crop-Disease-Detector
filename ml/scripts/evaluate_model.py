
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "ml" / "data" / "processed" / "tomato"
MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "tomato_disease_model.keras"
METADATA_PATH = PROJECT_ROOT / "backend" / "models" / "model_metadata.json"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not (DATA_DIR / "val").exists():
        raise FileNotFoundError(f"Validation folder not found: {DATA_DIR / 'val'}")

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    class_names = metadata["class_names"]

    val_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR / "val",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        class_names=class_names,
        shuffle=False,
    )

    model = keras.models.load_model(MODEL_PATH)

    true_labels = np.concatenate(
        [labels.numpy() for _, labels in val_ds],
        axis=0,
    )

    probabilities = model.predict(val_ds, verbose=1)
    predicted_labels = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(true_labels, predicted_labels)

    print("\n" + "=" * 60)
    print("AGRIVISION AI - VALIDATION EVALUATION")
    print("=" * 60)
    print(f"Validation images: {len(true_labels)}")
    print(f"Validation accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=list(range(len(class_names))),
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
    )

    print("Confusion Matrix:")
    print("Rows = Actual classes; Columns = Predicted classes")
    print("Class order:", class_names)
    print(matrix)

    results = {
        "evaluation_type": "validation",
        "validation_images": int(len(true_labels)),
        "accuracy_percent": round(float(accuracy) * 100, 2),
        "class_names": class_names,
        "classification_report": classification_report(
            true_labels,
            predicted_labels,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": matrix.tolist(),
    }

    output_path = PROJECT_ROOT / "ml" / "validation_evaluation.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
