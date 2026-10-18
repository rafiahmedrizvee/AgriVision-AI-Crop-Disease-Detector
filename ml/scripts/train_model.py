
from pathlib import Path
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "ml" / "data" / "processed" / "tomato"
MODEL_DIR = PROJECT_ROOT / "backend" / "models"

IMG_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 5
SEED = 42


def main():
    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"

    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            "Train or validation folder is missing. "
            f"Expected: {train_dir} and {val_dir}"
        )

    if not (DATA_DIR / "test").is_dir():
        raise FileNotFoundError(
            f"Test folder not found: {DATA_DIR / 'test'}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=SEED,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    class_names = train_ds.class_names

    if class_names != val_ds.class_names:
        raise ValueError(
            "Training and validation classes do not match."
        )

    if len(class_names) != 4:
        raise ValueError(
            f"Expected 4 classes, found {len(class_names)}: {class_names}"
        )

    train_count = sum(
        1 for class_name in class_names
        for path in (train_dir / class_name).iterdir()
        if path.is_file()
    )

    val_count = sum(
        1 for class_name in class_names
        for path in (val_dir / class_name).iterdir()
        if path.is_file()
    )

    test_count = sum(
        1 for class_name in class_names
        for path in (DATA_DIR / "test" / class_name).iterdir()
        if path.is_file()
    )

    print("\nClasses:", class_names)
    print("TensorFlow:", tf.__version__)
    print(f"Train images: {train_count}")
    print(f"Validation images: {val_count}")
    print(f"Test images reserved for final evaluation: {test_count}")

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*IMG_SIZE, 3))

    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1,
    )(inputs)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(
        len(class_names),
        activation="softmax",
    )(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    model_path = MODEL_DIR / "tomato_disease_model.keras"
    best_model_path = MODEL_DIR / "tomato_disease_best.keras"

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Save the best validation model, not necessarily the final epoch.
    best_model = keras.models.load_model(best_model_path)
    best_model.save(model_path)

    best_epoch = (
        int(min(
            range(len(history.history["val_loss"])),
            key=lambda i: history.history["val_loss"][i],
        )) + 1
    )

    metadata = {
        "class_names": class_names,
        "image_size": list(IMG_SIZE),
        "model_type": "MobileNetV2",
        "training_images": train_count,
        "validation_images": val_count,
        "test_images": test_count,
        "epochs_requested": EPOCHS,
        "epochs_completed": len(history.history["loss"]),
        "best_epoch": best_epoch,
        "best_val_accuracy": float(
            history.history["val_accuracy"][best_epoch - 1]
        ),
        "best_val_loss": float(
            history.history["val_loss"][best_epoch - 1]
        ),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "test_evaluated": False,
        "random_seed": SEED,
    }

    metadata_path = MODEL_DIR / "model_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nTraining finished!")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {metadata['best_val_accuracy']:.4f}")
    print(f"Model saved at: {model_path}")
    print(f"Metadata saved at: {metadata_path}")
    print("The test dataset has NOT been used for training or evaluation yet.")


if __name__ == "__main__":
    main()
