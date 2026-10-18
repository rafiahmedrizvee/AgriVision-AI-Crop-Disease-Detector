
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "PlantVillage" / "PlantVillage"
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed" / "tomato"

CLASSES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_names(folder):
    if not folder.exists():
        return set()

    return {
        file.name.lower()
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    }


def main():
    print("Checking raw dataset and processed split overlap...\n")

    for class_name in CLASSES:
        raw_folder = RAW_DIR / class_name
        train_folder = PROCESSED_DIR / "train" / class_name
        val_folder = PROCESSED_DIR / "val" / class_name

        raw_names = image_names(raw_folder)
        train_names = image_names(train_folder)
        val_names = image_names(val_folder)

        train_val_overlap = train_names & val_names
        train_raw_missing = train_names - raw_names
        val_raw_missing = val_names - raw_names

        print(f"Class: {class_name}")
        print(f"  Raw images: {len(raw_names)}")
        print(f"  Train images: {len(train_names)}")
        print(f"  Validation images: {len(val_names)}")
        print(f"  Same filenames in train and val: {len(train_val_overlap)}")
        print(f"  Train filenames missing from raw: {len(train_raw_missing)}")
        print(f"  Validation filenames missing from raw: {len(val_raw_missing)}")
        print()

    print("Note: filename matching does not detect renamed or duplicate image content.")


if __name__ == "__main__":
    main()
