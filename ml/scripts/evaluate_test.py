
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
MODEL_DIR = PROJECT_ROOT / "backend" / "models"

MODEL_PATH = MODEL_DIR / "tomato_disease_model.keras"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

IMG_SIZE = (160, 160)
BATCH_SIZE = 16


def main():
    test_dir = DATA_DIR / "test"

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test dataset not found: {test_dir}")

    print("Loading trained model...")
    model = keras.models.load_model(MODEL_PATH)

    print("Loading independent test dataset...")
    test_ds = keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    class_names = test_ds.class_names

    if len(class_names) != model.output_shape[-1]:
        raise ValueError(
            f"Model has {model.output_shape[-1]} outputs, "
            f"but dataset has {len(class_names)} classes."
        )

    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    # Predict each test image.
    y_true = np.concatenate([
        labels.numpy()
        for _, labels in test_ds
    ])

    probabilities = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    print("\n========== TEST RESULTS ==========")
    print(f"Test images: {len(y_true)}")
    print(f"Test accuracy: {accuracy:.4%}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(matrix)

    results = {
        "model": MODEL_PATH.name,
        "test_images": int(len(y_true)),
        "class_names": class_names,
        "test_accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    results_path = PROJECT_ROOT / "ml" / "test_evaluation.json"

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    # Update metadata only after successful independent evaluation.
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            metadata = json.load(file)
    else:
        metadata = {}

    metadata["test_evaluated"] = True
    metadata["test_accuracy"] = float(accuracy)
    metadata["test_images"] = int(len(y_true))
    metadata["test_results_file"] = "ml/test_evaluation.json"

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"\nDetailed results saved to: {results_path}")
    print(f"Model metadata updated: {METADATA_PATH}")


if __name__ == "__main__":
    main()
