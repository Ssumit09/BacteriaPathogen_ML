import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------
# CONFIG
# ----------------------------
IMG_SIZE = (224, 224)  # image resize target
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

# paths
BASE_DIR = Path("metadata")
ENCODER_DIR = BASE_DIR / "encoders"
SPLIT_DIR = BASE_DIR / "splits"

# Load encoders
encoder: OneHotEncoder = joblib.load(ENCODER_DIR / "onehot_encoder.pkl")
scaler: StandardScaler = joblib.load(ENCODER_DIR / "scaler.pkl")

# Categorical and numeric columns (same as before)
categorical_cols = ['agar', 'species']
numeric_cols = ['time_hr']

# ----------------------------
# Function to encode metadata
# ----------------------------
def encode_metadata(row):
    cat_values = np.array([[row['agar'], row['species']]])
    num_values = np.array([[row['time_hr']]])

    encoded_cats = encoder.transform(cat_values)
    scaled_nums = scaler.transform(num_values)

    metadata_vector = np.concatenate([encoded_cats[0], scaled_nums[0]], axis=0)
    return metadata_vector.astype(np.float32)

# ----------------------------
# Function to load image
# ----------------------------
def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

# ----------------------------
# Convert a DataFrame to tf.data.Dataset
# ----------------------------
def create_tf_dataset(csv_path):
    df = pd.read_csv(csv_path)
    image_paths = df['filepath'].values
    labels = df['label_bpseudomallei'].values.astype(np.float32)
    
    # Pre-encode all metadata upfront
    metadata_encoded = np.stack(df.apply(encode_metadata, axis=1).values)
    
    dataset = tf.data.Dataset.from_tensor_slices(((image_paths, metadata_encoded), labels))

    def load_and_preprocess(features, label):
        image_path, metadata=features
        image = load_image(image_path)
        return (image, metadata), label

    dataset = dataset.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(df))
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return dataset

# ----------------------------
# Create train, val, test datasets
# ----------------------------
train_ds = create_tf_dataset(SPLIT_DIR / "train.csv")
val_ds = create_tf_dataset(SPLIT_DIR / "val.csv")
test_ds = create_tf_dataset(SPLIT_DIR / "test.csv")

print("✅ TensorFlow Datasets created successfully:")
print(f"Train batches: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

# Optional check (inspect one sample)
for (img, meta), label in train_ds.take(1):
    print("\n🧪 Sample shapes:")
    print("  Image:", img.shape)
    print("  Metadata vector:", meta.shape)
    print("  Label:", label.shape)
