
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

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
RANDOM_SEED = 42


def main():
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(
            f"Source folder not found: {SOURCE_DIR}"
        )

    # Never overwrite an existing dataset automatically.
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"\nOutput folder is not empty:\n{OUTPUT_DIR}\n\n"
            "Rename or move the existing tomato folder first.\n"
            "Your backup is tomato_backup."
        )

    # Validate every class before copying anything.
    class_images = {}

    for class_name in CLASSES:
        class_dir = SOURCE_DIR / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Class folder not found: {class_dir}"
            )

        images = sorted(
            file
            for file in class_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        if len(images) < 10:
            raise ValueError(
                f"Not enough images in {class_name}: {len(images)}"
            )

        class_images[class_name] = images

    total_train = 0
    total_val = 0
    total_test = 0

    try:
        for class_name, images in class_images.items():
            # Reserve 15% for testing.
            train_val_images, test_images = train_test_split(
                images,
                test_size=TEST_SIZE,
                random_state=RANDOM_SEED,
                shuffle=True,
            )

            # Split the remaining 85% so validation is 15%
            # of the complete dataset, not 15% of 85%.
            relative_val_size = VALIDATION_SIZE / (1 - TEST_SIZE)

            train_images, val_images = train_test_split(
                train_val_images,
                test_size=relative_val_size,
                random_state=RANDOM_SEED,
                shuffle=True,
            )

            splits = {
                "train": train_images,
                "val": val_images,
                "test": test_images,
            }

            for split_name, split_images in splits.items():
                destination = (
                    OUTPUT_DIR / split_name / class_name
                )
                destination.mkdir(parents=True, exist_ok=True)

                for image_path in split_images:
                    shutil.copy2(
                        image_path,
                        destination / image_path.name,
                    )

            total_train += len(train_images)
            total_val += len(val_images)
            total_test += len(test_images)

            print(
                f"{class_name}: "
                f"train={len(train_images)}, "
                f"validation={len(val_images)}, "
                f"test={len(test_images)}"
            )

    except Exception:
        # Remove incomplete output if copying fails.
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)
        raise

    total = total_train + total_val + total_test

    print("\nDataset preparation complete!")
    print(f"Total images: {total}")
    print(f"Training images: {total_train} ({total_train / total:.1%})")
    print(f"Validation images: {total_val} ({total_val / total:.1%})")
    print(f"Test images: {total_test} ({total_test / total:.1%})")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
