
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

SOURCE_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "PlantVillage"
    / "PlantVillage"
)

OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "tomato"
)

CLASSES = [
    "Tomato_healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALIDATION_SIZE = 0.20
RANDOM_SEED = 42


def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source folder not found: {SOURCE_DIR}")

    # Avoid accidentally mixing new and old split files.
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty: {OUTPUT_DIR}\n"
            "Move or rename this folder before running again."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_train = 0
    total_val = 0

    for class_name in CLASSES:
        class_dir = SOURCE_DIR / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")

        images = sorted(
            file for file in class_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        if len(images) < 2:
            raise ValueError(f"Not enough images in {class_name}")

        train_images, val_images = train_test_split(
            images,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_SEED,
            shuffle=True,
        )

        for split_name, split_images in [
            ("train", train_images),
            ("val", val_images),
        ]:
            destination = OUTPUT_DIR / split_name / class_name
            destination.mkdir(parents=True, exist_ok=True)

            for image_path in split_images:
                shutil.copy2(image_path, destination / image_path.name)

        total_train += len(train_images)
        total_val += len(val_images)

        print(
            f"{class_name}: "
            f"train={len(train_images)}, validation={len(val_images)}"
        )

    print("\nDataset preparation complete!")
    print(f"Training images: {total_train}")
    print(f"Validation images: {total_val}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
