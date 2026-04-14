"""Training script for the multi-disease chest X-ray classifier."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf


CLASS_NAMES = ["COVID", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]
IMAGE_SIZE = (224, 224)


def build_model() -> tf.keras.Model:
    """Build the ResNet50 transfer learning model used for training."""
    try:
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(224, 224, 3),
        )
    except Exception:
        # Training can still run with randomly initialized weights if pretrained weights are unavailable.
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights=None,
            input_shape=(224, 224, 3),
        )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Lambda(tf.keras.applications.resnet.preprocess_input)(inputs)
    x = base_model(x, training=False)
    # Keep a connected 4D activation tensor for Grad-CAM in Keras 3.
    x = tf.keras.layers.Lambda(lambda t: t, name="last_conv_map")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="chest_xray_resnet50")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_data_generators(data_dir: str, batch_size: int):
    """Create training and validation generators with augmentation."""
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=12,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.08,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2,
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        classes=CLASS_NAMES,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        classes=CLASS_NAMES,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    return train_generator, validation_generator


def plot_history(history: tf.keras.callbacks.History, output_path: str) -> None:
    """Save a simple training accuracy/loss plot for the project report."""
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history.get("accuracy", []), label="Train Accuracy")
    plt.plot(history.history.get("val_accuracy", []), label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history.get("loss", []), label="Train Loss")
    plt.plot(history.history.get("val_loss", []), label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_dummy_model(model_path: str) -> None:
    """Save a randomly initialized model when no training data is available."""
    model = build_model()
    model.save(model_path)
    print(f"No dataset found. Saved an untrained model to {model_path}.")


def train_model(data_dir: str, model_path: str, epochs: int, batch_size: int) -> None:
    """Train the classifier and save the final model and report plots."""
    if not os.path.isdir(data_dir):
        save_dummy_model(model_path)
        return

    train_generator, validation_generator = create_data_generators(data_dir, batch_size)
    if train_generator.samples == 0 or validation_generator.samples == 0:
        save_dummy_model(model_path)
        return

    model = build_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
        ),
    ]

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=callbacks,
    )

    model.save(model_path)
    plot_history(history, os.path.join(Path(model_path).parent, "training_history.png"))
    print(f"Training complete. Model saved to {model_path}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the chest X-ray classifier.")
    parser.add_argument("--data-dir", default="dataset", help="Path to dataset root folder.")
    parser.add_argument("--model-path", default=os.path.join("model", "model.h5"), help="Output model path.")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    train_model(args.data_dir, args.model_path, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()