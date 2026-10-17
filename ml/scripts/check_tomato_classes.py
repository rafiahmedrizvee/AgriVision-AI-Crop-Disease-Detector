
from pathlib import Path

DATASET_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "PlantVillage"
    / "PlantVillage"
)

SELECTED_CLASSES = [
    "Tomato_healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    print(f"Dataset directory: {DATASET_DIR}\n")

    if not DATASET_DIR.exists():
        print("ERROR: Dataset directory not found.")
        print("Please check the folder path.")
        return

    total_images = 0

    for class_name in SELECTED_CLASSES:
        class_dir = DATASET_DIR / class_name

        if not class_dir.is_dir():
            print(f"MISSING: {class_name}")
            continue

        count = sum(
            1
            for file in class_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        total_images += count
        print(f"{class_name}: {count} images")

    print(f"\nTotal images in selected classes: {total_images}")


if __name__ == "__main__":
    main()
