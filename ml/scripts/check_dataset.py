
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    print(f"Dataset search location: {RAW_DIR}")

    if not RAW_DIR.exists():
        print("ERROR: Dataset raw folder was not found.")
        return

    class_folders = []

    for folder in RAW_DIR.rglob("*"):
        if not folder.is_dir():
            continue

        image_count = sum(
            1
            for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        if image_count > 0 and (
            "tomato" in folder.name.lower()
            or folder.parent.name.lower() == "tomato"
        ):
            class_folders.append((folder.name, image_count))

    if not class_folders:
        print("No tomato image folders found.")
        print("Check the extracted dataset folder structure.")
        return

    print("\nTomato image folders found:\n")

    total = 0
    for name, count in sorted(class_folders):
        print(f"{name}: {count} images")
        total += count

    print(f"\nTotal images in listed folders: {total}")
    print(f"Number of folders: {len(class_folders)}")

if __name__ == "__main__":
    main()
