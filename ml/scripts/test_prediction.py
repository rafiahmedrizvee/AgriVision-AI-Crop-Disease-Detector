
from pathlib import Path
import numpy as np
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "tomato_disease_model.keras"
VAL_DIR = PROJECT_ROOT / "ml" / "data" / "processed" / "tomato" / "val"

IMAGE_SIZE = (160, 160)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not VAL_DIR.exists():
        raise FileNotFoundError(f"Validation folder not found: {VAL_DIR}")

    print("Loading trained model...")
    model = keras.models.load_model(MODEL_PATH)

    # Match the alphabetically sorted class order used during training.
    class_names = sorted(
        folder.name for folder in VAL_DIR.iterdir()
        if folder.is_dir()
    )

    if len(class_names) != model.output_shape[-1]:
        raise ValueError("Class folders do not match the model output.")

    print("\nTesting one validation image from each class:\n")

    for actual_class in class_names:
        class_dir = VAL_DIR / actual_class
        images = sorted(
            path for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not images:
            print(f"SKIPPED: No images found in {actual_class}")
            continue

        image_path = images[0]

        image = keras.utils.load_img(
            image_path,
            target_size=IMAGE_SIZE,
            color_mode="rgb",
        )
        image_array = keras.utils.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)

        probabilities = model.predict(image_array, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))
        predicted_class = class_names[predicted_index]
        confidence = float(probabilities[predicted_index]) * 100

        print(f"Image class: {actual_class}")
        print(f"Predicted:   {predicted_class}")
        print(f"Confidence:  {confidence:.2f}%")
        print(f"Result:      {'MATCH' if predicted_class == actual_class else 'MISMATCH'}")
        print("-" * 45)


if __name__ == "__main__":
    main()
