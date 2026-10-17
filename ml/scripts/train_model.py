
from pathlib import Path
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "ml" / "data" / "processed" / "tomato"
MODEL_DIR = PROJECT_ROOT / "backend" / "models"

IMG_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 5
SEED = 42


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR / "train",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=SEED,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR / "val",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    class_names = train_ds.class_names

    if class_names != val_ds.class_names:
        raise ValueError("Training and validation classes do not match.")

    print("\nClasses:", class_names)
    print("TensorFlow:", tf.__version__)

    # Improve input pipeline performance.
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # Pretrained image feature extractor.
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

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "tomato_disease_model.keras"),
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

    model_path = MODEL_DIR / "tomato_disease_model.keras"
    model.save(model_path)

    metadata = {
        "class_names": class_names,
        "image_size": list(IMG_SIZE),
        "model_type": "MobileNetV2",
        "training_images": 5300,
        "validation_images": 1327,
        "epochs_requested": EPOCHS,
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
    }

    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nTraining finished!")
    print(f"Model saved at: {model_path}")
    print(f"Metadata saved at: {MODEL_DIR / 'model_metadata.json'}")


if __name__ == "__main__":
    main()
