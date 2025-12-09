# """
# grad_cam_batch.py
# Generates and saves Grad-CAM visualizations for all images in the test dataset.
# """

# import tensorflow as tf
# import numpy as np
# import pandas as pd
# import cv2
# from pathlib import Path
# import joblib
# from data_loader import encode_metadata, load_image
# from tensorflow.keras.models import load_model

# # ----------------------------
# # CONFIGURATION
# # ----------------------------
# MODEL_PATH = Path("models/final_finetuned_model.keras")  # ✅ Use folder, not .keras
# TEST_CSV = Path("metadata/splits/test.csv")
# SAVE_DIR = Path("outputs/gradcam_test/")
# SAVE_DIR.mkdir(parents=True, exist_ok=True)
# RAW_IMG_DIR = Path("data/raw")  # 👈 directory where your actual images live

# IMG_SIZE = (224, 224)

# # ----------------------------
# # LOAD MODEL AND FIND LAST CONV LAYER
# # ----------------------------
# print("📂 Loading fine-tuned model...")
# model = load_model(MODEL_PATH, compile=False)

# last_conv_layer_name = None
# for layer in reversed(model.layers):
#     if isinstance(layer, tf.keras.layers.Conv2D):
#         last_conv_layer_name = layer.name
#         break
# if not last_conv_layer_name:
#     raise ValueError("Could not automatically find a Conv2D layer in the model.")
# print(f"✅ Using layer '{last_conv_layer_name}' for Grad-CAM.")

# # ----------------------------
# # GRAD-CAM FUNCTION
# # ----------------------------
# def make_gradcam_heatmap(model, image_array, metadata_vector, last_conv_layer_name):
#     grad_model = tf.keras.models.Model(
#         [model.inputs],
#         [model.get_layer(last_conv_layer_name).output, model.output]
#     )

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model([image_array, metadata_vector])
#         pred_index = tf.argmax(predictions[0])
#         class_channel = predictions[:, pred_index]

#     grads = tape.gradient(class_channel, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#     conv_outputs = conv_outputs[0]
#     heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
#     heatmap = np.maximum(heatmap, 0)
#     if tf.reduce_max(heatmap) > 0:
#         heatmap /= tf.reduce_max(heatmap)
#     return heatmap.numpy(), predictions[0][0]

# # ----------------------------
# # OVERLAY HEATMAP ON IMAGE
# # ----------------------------
# def overlay_heatmap(heatmap, image_path, alpha=0.4):
#     img = cv2.imread(str(image_path))
#     if img is None:
#         raise ValueError(f"❌ Could not load image: {image_path}")
#     img = cv2.resize(img, IMG_SIZE)

#     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     overlay = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
#     return overlay

# # ----------------------------
# # PROCESS ALL TEST IMAGES
# # ----------------------------
# def process_all_test_images():
#     df = pd.read_csv(TEST_CSV)
#     results = []

#     for _, row in df.iterrows():
#         img_path = RAW_IMG_DIR / Path(row["filepath"]).name
#         if not img_path.exists():
#             raise FileNotFoundError(f"Image not found: {img_path}")

#         agar, species, time_hr = row["agar"], row["species"], row["time_hr"]
#         label = int(row["label_bpseudomallei"])

#         try:
#             img_tensor = tf.expand_dims(load_image(str(img_path)), axis=0)
#             meta_vector = np.expand_dims(encode_metadata(row), axis=0)

#             heatmap, confidence = make_gradcam_heatmap(model, img_tensor, meta_vector, last_conv_layer_name)

#             # Debug info
#             print(f"Processing {img_path.name}: heatmap range = [{heatmap.min():.3f}, {heatmap.max():.3f}]")

#             overlay = overlay_heatmap(heatmap, img_path)
#             save_path = SAVE_DIR / f"{img_path.stem}_GradCAM.jpg"
#             success = cv2.imwrite(str(save_path), overlay)

#             if success:
#                 print(f"✅ Saved Grad-CAM for {img_path.name} (Confidence: {confidence:.3f})")
#             else:
#                 print(f"⚠️ Failed to save Grad-CAM image for {img_path.name}")

#             results.append({
#                 "filename": img_path.name,
#                 "predicted_confidence": float(confidence),
#                 "true_label": label,
#                 "predicted_label": 1 if confidence >= 0.5 else 0,
#                 "agar": agar,
#                 "species": species,
#                 "time_hr": time_hr
#             })

#         except Exception as e:
#             print(f"❌ Error processing {img_path.name}: {e}")

#     summary_df = pd.DataFrame(results)
#     summary_path = SAVE_DIR / "gradcam_summary.csv"
#     summary_df.to_csv(summary_path, index=False)
#     print(f"\n📊 Summary saved to: {summary_path}")
#     print(f"🖼️ Heatmaps saved in: {SAVE_DIR}")

# # ----------------------------
# # MAIN EXECUTION
# # ----------------------------
# if __name__ == "__main__":
#     process_all_test_images()




# ----------------------------

"""
grad_cam_batch.py
Generates and saves Grad-CAM visualizations for all images in the validation dataset.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import joblib
import warnings
from data_loader import encode_metadata, load_image
from tensorflow.keras.models import load_model

# Suppress warnings
warnings.filterwarnings('ignore')

# ----------------------------
# CONFIGURATION
# ----------------------------
MODEL_PATH = Path("models/final_finetuned_model.keras")
VAL_CSV = Path("metadata/splits/val.csv")
SAVE_DIR = Path("outputs/gradcam_val/")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
RAW_IMG_DIR = Path("data/raw")

IMG_SIZE = (224, 224)

# ----------------------------
# LOAD MODEL AND FIND LAST CONV LAYER
# ----------------------------
print("📂 Loading fine-tuned model...")
model = load_model(str(MODEL_PATH), compile=False)

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
def make_gradcam_heatmap(model, image_tensor, metadata_tensor, last_conv_layer_name):
    """
    Compute Grad-CAM heatmap.
    
    Args:
        model: Keras model
        image_tensor: (1, 224, 224, 3) float32 tensor
        metadata_tensor: (1, n_features) float32 tensor
        last_conv_layer_name: name of Conv2D layer
    
    Returns:
        heatmap: (H, W) normalized numpy array
        prediction: scalar prediction value
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([image_tensor, metadata_tensor], training=False)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    
    if tf.reduce_max(heatmap) > 0:
        heatmap /= tf.reduce_max(heatmap)
    
    return heatmap.numpy(), float(predictions[0][0].numpy())

# ----------------------------
# OVERLAY HEATMAP ON IMAGE
# ----------------------------
def overlay_heatmap(heatmap, image_path, alpha=0.4):
    """Overlay Grad-CAM heatmap on original image."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"❌ Could not load image: {image_path}")
    
    img = cv2.resize(img, IMG_SIZE)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
    
    return overlay

# ----------------------------
# PROCESS ALL VALIDATION IMAGES
# ----------------------------
def process_all_validation_images():
    """Process all images in validation set and save Grad-CAM visualizations."""
    df = pd.read_csv(VAL_CSV)
    results = []
    success_count = 0
    error_count = 0

    print(f"\n🔄 Processing {len(df)} validation images...\n")

    for idx, row in df.iterrows():
        img_path = RAW_IMG_DIR / Path(row["filepath"]).name
        
        if not img_path.exists():
            print(f"⚠️  Image not found: {img_path}")
            error_count += 1
            continue

        agar, species, time_hr = row["agar"], row["species"], row["time_hr"]
        label = int(row["label_bpseudomallei"])

        try:
            # Load and prepare image as tensor
            img_array = load_image(str(img_path)).numpy()
            img_tensor = tf.expand_dims(img_array, axis=0)
            
            # Load and prepare metadata as tensor
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                meta_array = encode_metadata(row)
            meta_tensor = tf.expand_dims(meta_array, axis=0)

            # Ensure both are float32 tensors
            img_tensor = tf.cast(img_tensor, tf.float32)
            meta_tensor = tf.cast(meta_tensor, tf.float32)

            # Compute Grad-CAM
            heatmap, confidence = make_gradcam_heatmap(
                model, img_tensor, meta_tensor, last_conv_layer_name
            )

            # Overlay and save
            overlay = overlay_heatmap(heatmap, img_path)
            save_path = SAVE_DIR / f"{img_path.stem}_GradCAM.jpg"
            success = cv2.imwrite(str(save_path), overlay)

            if success:
                print(f"✅ [{idx+1}/{len(df)}] {img_path.name} (Conf: {confidence:.3f})")
                success_count += 1
            else:
                print(f"❌ [{idx+1}/{len(df)}] Failed to save: {img_path.name}")
                error_count += 1

            results.append({
                "filename": img_path.name,
                "predicted_confidence": float(confidence),
                "true_label": label,
                "predicted_label": 1 if confidence >= 0.5 else 0,
                "agar": agar,
                "species": species,
                "time_hr": time_hr
            })

        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {str(e)}")
            error_count += 1

    # Save summary
    if results:
        summary_df = pd.DataFrame(results)
        summary_path = SAVE_DIR / "gradcam_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n📊 Summary saved to: {summary_path}")
    
    print(f"\n✅ Processing complete:")
    print(f"   Successful: {success_count}")
    print(f"   Errors: {error_count}")
    print(f"📁 Heatmaps saved in: {SAVE_DIR}")

# ----------------------------
# MAIN EXECUTION
# ----------------------------
if __name__ == "__main__":
    process_all_validation_images()