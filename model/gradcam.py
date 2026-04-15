"""Grad-CAM utilities for chest X-ray explainability."""

from __future__ import annotations

import cv2
import numpy as np


def get_tensorflow():
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorFlow is not installed in the active Python environment. "
            "Activate the project .venv and run python app.py from there."
        ) from exc

    return tf


def get_last_conv_layer_name(model) -> str:
    """Automatically find the last convolutional layer in a model."""
    tf = get_tensorflow()
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

        try:
            output_shape = layer.output.shape
            if len(output_shape) == 4:
                return layer.name
        except Exception:
            continue

    raise ValueError("No convolutional layer found for Grad-CAM generation.")


def make_gradcam_heatmap(
    image_array: np.ndarray,
    model,
    last_conv_layer_name: str | None = None,
    class_index: int | None = None,
) -> np.ndarray:
    """Create a normalized Grad-CAM heatmap for a single image batch."""
    tf = get_tensorflow()
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer_name(model)

    last_conv_layer = model.get_layer(last_conv_layer_name)
    # Use model.inputs directly; wrapping it in an extra list can create a nested input signature
    # that breaks Functional.call when passing a normal image batch.
    grad_model = tf.keras.models.Model(model.inputs, [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([image_array])
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        class_channel = predictions[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_value + tf.keras.backend.epsilon())
    return heatmap.numpy()


def overlay_heatmap(
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    output_path: str,
    alpha: float = 0.42,
) -> str:
    """Overlay the heatmap on the original image and save the result."""
    colored_heatmap = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(colored_heatmap, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    colored_heatmap = cv2.resize(colored_heatmap, (original_rgb.shape[1], original_rgb.shape[0]))

    overlay = cv2.addWeighted(original_rgb, 1 - alpha, colored_heatmap, alpha, 0)
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return output_path


def save_heatmap_image(
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    output_path: str,
) -> str:
    """Persist only the colorized heatmap to disk (without blending)."""
    colored_heatmap = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(colored_heatmap, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    colored_heatmap = cv2.resize(colored_heatmap, (original_rgb.shape[1], original_rgb.shape[0]))
    cv2.imwrite(output_path, cv2.cvtColor(colored_heatmap, cv2.COLOR_RGB2BGR))
    return output_path


def generate_gradcam_visuals(
    model,
    image_array: np.ndarray,
    original_rgb: np.ndarray,
    heatmap_output_path: str,
    overlay_output_path: str,
    class_index: int | None = None,
    last_conv_layer_name: str | None = None,
) -> tuple[str, str]:
    """Generate and persist both heatmap-only and overlay Grad-CAM images."""
    heatmap = make_gradcam_heatmap(
        image_array=image_array,
        model=model,
        last_conv_layer_name=last_conv_layer_name,
        class_index=class_index,
    )
    save_heatmap_image(original_rgb=original_rgb, heatmap=heatmap, output_path=heatmap_output_path)
    overlay_heatmap(original_rgb=original_rgb, heatmap=heatmap, output_path=overlay_output_path)
    return heatmap_output_path, overlay_output_path