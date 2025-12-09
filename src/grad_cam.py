"""
grad_cam.py

Generates Grad-CAM heatmaps for the fine-tuned model to visualize
which parts of the Petri dish image influenced the prediction.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import joblib
import pandas as pd
from data_loader import encode_metadata, load_image
from tensorflow.keras.models import load_model

# ----------------------------
# CONFIGURATION
# ----------------------------
MODEL_PATH = Path("models/final_finetuned_model.keras")
ENCODER_DIR = Path("metadata/encoders")
IMG_SIZE = (224, 224)

# ----------------------------
# LOAD MODEL AND ENCODERS
# ----------------------------
print("📂 Loading fine-tuned model...")
model = load_model(MODEL_PATH)

# Find the last convolutional layer for Grad-CAM
last_conv_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer_name = layer.name
        break
if not last_conv_layer_name:
    raise ValueError("Could not automatically find a Conv2D layer in the model.")

print(f"✅ Using layer '{last_conv_layer_name}' for Grad-CAM.")

# ----------------------------
# GRAD-CAM FUNCTION
# ----------------------------
def make_gradcam_heatmap(model, image_array, metadata_vector, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([image_array, metadata_vector])
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Compute gradients of top predicted class wrt conv outputs
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalize between 0 and 1 for visualization
    heatmap = np.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

# ----------------------------
# OVERLAY HEATMAP ON IMAGE
# ----------------------------
def overlay_heatmap(heatmap, image_path, alpha=0.4):
    img = cv2.imread(str(image_path))
    img = cv2.resize(img, IMG_SIZE)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    output = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
    return output

# ----------------------------
# RUN GRAD-CAM ON SAMPLE IMAGE
# ----------------------------
def run_gradcam_on_sample(image_path, agar, species, time_hr):
    print(f"\n🧫 Running Grad-CAM for: {image_path.name}")

    # Load and preprocess image
    img_tensor = tf.expand_dims(load_image(str(image_path)), axis=0)

    # Encode metadata
    meta_row = {"agar": agar, "species": species, "time_hr": time_hr}
    meta_vector = np.expand_dims(encode_metadata(meta_row), axis=0)

    # Generate heatmap
    heatmap = make_gradcam_heatmap(model, img_tensor, meta_vector, last_conv_layer_name)

    # Overlay heatmap
    overlay = overlay_heatmap(heatmap, image_path)

    # Show
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM Heatmap Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# ----------------------------
# MAIN EXECUTION
# ----------------------------
if __name__ == "__main__":
    sample_image = Path("data/raw/3-Ashdown_Agar_PseudomonasAeruginosa_24HR.png")

    run_gradcam_on_sample(
        image_path=sample_image,
        agar="Ashdown",
        species="PseudomonasAeruginosa",
        time_hr=24
    )
