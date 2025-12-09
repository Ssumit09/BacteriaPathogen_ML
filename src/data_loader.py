# """
# data_loader.py

# Loads both images and metadata, applies TensorFlow native data augmentation
# to the training set only, and returns tf.data.Dataset objects for train/val/test.
# """

# import tensorflow as tf
# import pandas as pd
# import numpy as np
# from pathlib import Path
# import joblib
# from sklearn.preprocessing import OneHotEncoder, StandardScaler

# # ----------------------------
# # CONFIGURATION
# # ----------------------------
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 16
# AUTOTUNE = tf.data.AUTOTUNE

# BASE_DIR = Path("metadata")
# ENCODER_DIR = BASE_DIR / "encoders"
# SPLIT_DIR = BASE_DIR / "splits"

# # Load encoders
# encoder: OneHotEncoder = joblib.load(ENCODER_DIR / "onehot_encoder.pkl")
# scaler: StandardScaler = joblib.load(ENCODER_DIR / "scaler.pkl")

# categorical_cols = ['agar', 'species']
# numeric_cols = ['time_hr']

# # ----------------------------
# # 1️⃣ Data Augmentation Layer (applied only to train)
# # ----------------------------
# data_augmentation = tf.keras.Sequential([
#     tf.keras.layers.RandomFlip("horizontal_and_vertical"),
#     tf.keras.layers.RandomRotation(0.2),
#     tf.keras.layers.RandomZoom(0.1),
#     tf.keras.layers.RandomBrightness(0.2),
# ], name="data_augmentation")

# # ----------------------------
# # 2️⃣ Metadata Encoding
# # ----------------------------
# def encode_metadata(row):
#     cat_values = np.array([[row['agar'], row['species']]])
#     num_values = np.array([[row['time_hr']]])

#     encoded_cats = encoder.transform(cat_values)
#     scaled_nums = scaler.transform(num_values)

#     metadata_vector = np.concatenate([encoded_cats[0], scaled_nums[0]], axis=0)
#     return metadata_vector.astype(np.float32)

# # ----------------------------
# # 3️⃣ Image Loading
# # ----------------------------
# def load_image(img_path):
#     img = tf.io.read_file(img_path)
#     img = tf.image.decode_jpeg(img, channels=3)
#     img = tf.image.resize(img, IMG_SIZE)
#     img = tf.cast(img, tf.float32) / 255.0
#     return img

# # ----------------------------
# # 4️⃣ Dataset Creation Function
# # ----------------------------
# def create_tf_dataset(csv_path, training=False):
#     df = pd.read_csv(csv_path)
#     image_paths = df['filepath'].values
#     labels = df['label_bpseudomallei'].values.astype(np.float32)
    
#     # Pre-encode all metadata upfront
#     metadata_encoded = np.stack(df.apply(encode_metadata, axis=1).values)

#     dataset = tf.data.Dataset.from_tensor_slices(((image_paths, metadata_encoded), labels))

#     def load_and_preprocess(inputs,label):
#         image_path,metadata=inputs
#         image = load_image(image_path)
#         if training:
#             image = data_augmentation(image)  # Apply augmentation only in training
#         return (image, metadata), label

#     dataset = dataset.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    
#     if training:
#         dataset = dataset.shuffle(buffer_size=len(df))
    
#     dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
#     return dataset

# # ----------------------------
# # 5️⃣ Convenience Function — Create All Datasets
# # ----------------------------
# def get_datasets():
#     train_ds = create_tf_dataset(SPLIT_DIR / "train.csv", training=True)
#     val_ds = create_tf_dataset(SPLIT_DIR / "val.csv", training=False)
#     test_ds = create_tf_dataset(SPLIT_DIR / "test.csv", training=False)
#     return train_ds, val_ds, test_ds

# # ----------------------------
# # Optional: quick test
# # ----------------------------
# if __name__ == "__main__":
#     train_ds, val_ds, test_ds = get_datasets()
#     print("✅ TensorFlow datasets ready:")
#     print(f"Train batches: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

#     for (img, meta), label in train_ds.take(1):
#         print("\n🧪 Sample batch shapes:")
#         print("  Image:", img.shape)
#         print("  Metadata:", meta.shape)
#         print("  Label:", label.shape)




# --------------------------------------------

"""
data_loader.py

Loads both images and metadata, applies TensorFlow native data augmentation
to the training set only, and returns tf.data.Dataset objects for train/val.
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore')

# ----------------------------
# CONFIGURATION
# ----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

BASE_DIR = Path("metadata")
ENCODER_DIR = BASE_DIR / "encoders"
SPLIT_DIR = BASE_DIR / "splits"

# Load encoders
encoder: OneHotEncoder = joblib.load(ENCODER_DIR / "onehot_encoder.pkl")
scaler: StandardScaler = joblib.load(ENCODER_DIR / "scaler.pkl")

categorical_cols = ['agar', 'species']
numeric_cols = ['time_hr']

# ----------------------------
# 1️⃣ Data Augmentation Layer (applied only to train)
# ----------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomBrightness(0.2),
], name="data_augmentation")

# ----------------------------
# 2️⃣ Metadata Encoding
# ----------------------------
def encode_metadata(row):
    """
    Convert categorical and numeric metadata to float32 vector.
    Suppresses warnings for missing feature names.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        cat_values = np.array([[row['agar'], row['species']]])
        num_values = np.array([[row['time_hr']]])

        encoded_cats = encoder.transform(cat_values)
        scaled_nums = scaler.transform(num_values)

        metadata_vector = np.concatenate([encoded_cats[0], scaled_nums[0]], axis=0)
    
    return metadata_vector.astype(np.float32)

# ----------------------------
# 3️⃣ Image Loading
# ----------------------------
def load_image(img_path):
    """Load and preprocess image to (224, 224, 3) float32 [0, 1]"""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

# ----------------------------
# 4️⃣ Dataset Creation Function
# ----------------------------
def create_tf_dataset(csv_path, training=False):
    """
    Create tf.data.Dataset from CSV with image paths and metadata.
    Returns: Dataset with ((image, metadata), label) tuples as tensors.
    """
    df = pd.read_csv(csv_path)
    image_paths = df['filepath'].values
    labels = df['label_bpseudomallei'].values.astype(np.float32)
    
    # Pre-encode all metadata upfront
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        metadata_encoded = np.stack(df.apply(encode_metadata, axis=1).values)

    dataset = tf.data.Dataset.from_tensor_slices(((image_paths, metadata_encoded), labels))

    def load_and_preprocess(inputs, label):
        image_path, metadata = inputs
        image = load_image(image_path)
        
        # Ensure metadata is tensor
        metadata = tf.convert_to_tensor(metadata, dtype=tf.float32)
        
        if training:
            image = data_augmentation(image)  # Apply augmentation only in training
        
        return (image, metadata), label

    dataset = dataset.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    
    if training:
        dataset = dataset.shuffle(buffer_size=len(df))
    
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return dataset

# ----------------------------
# 5️⃣ Convenience Function – Create All Datasets
# ----------------------------
def get_datasets():
    """Load train and validation datasets (80-20 split)"""
    train_ds = create_tf_dataset(SPLIT_DIR / "train.csv", training=True)
    val_ds = create_tf_dataset(SPLIT_DIR / "val.csv", training=False)
    return train_ds, val_ds

# ----------------------------
# Optional: quick test
# ----------------------------
if __name__ == "__main__":
    train_ds, val_ds = get_datasets()
    print("✅ TensorFlow datasets ready:")
    print(f"Train batches: {len(train_ds)}, Val: {len(val_ds)}")

    for (img, meta), label in train_ds.take(1):
        print("\n🧪 Sample batch shapes:")
        print("  Image:", img.shape)
        print("  Metadata:", meta.shape)
        print("  Label:", label.shape)