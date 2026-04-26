"""
data_loader_fixed.py - CORRECTED VERSION
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import warnings
from tensorflow.keras.applications.efficientnet import preprocess_input

warnings.filterwarnings('ignore')

# CONFIGURATION
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

BASE_DIR = Path("metadata")
ENCODER_DIR = BASE_DIR / "encoders"
SPLIT_DIR = BASE_DIR / "splits_fixed"
RAW_DATA_DIR = Path("data/raw")

# Load encoders
encoder = joblib.load(ENCODER_DIR / "onehot_encoder.pkl")
scaler = joblib.load(ENCODER_DIR / "scaler.pkl")

categorical_cols = ['agar', 'species']
numeric_cols = ['time_hr']

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomBrightness(0.3),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
], name="aggressive_augmentation")

# Encode metadata
def encode_metadata(row):
    """Convert metadata to float32 vector"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cat_values = np.array([[row['agar'], row['species']]])
        num_values = np.array([[row['time_hr']]])
        encoded_cats = encoder.transform(cat_values)
        scaled_nums = scaler.transform(num_values)
        metadata_vector = np.concatenate([encoded_cats[0], scaled_nums[0]], axis=0)
    return metadata_vector.astype(np.float32)

# Load image
def load_image(img_path):
    """Load and preprocess image"""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img

# Load and preprocess with augmentation
def load_and_preprocess(image_path, metadata, apply_augmentation):
    """Load image and optionally apply augmentation"""
    image = load_image(image_path)
    
    if apply_augmentation:
        image = data_augmentation(image, training=True)
    
    return (image, tf.cast(metadata, tf.float32))

# Create dataset
def create_tf_dataset(csv_path, training=False, shuffle=True):
    """Create tf.data.Dataset from CSV"""
    df = pd.read_csv(csv_path)
    
    # Build image paths
    image_paths = [str(RAW_DATA_DIR / filename) for filename in df['filename'].values]
    labels = df['label_bpseudomallei'].values.astype(np.float32)
    
    # Encode metadata
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        metadata_encoded = np.stack(df.apply(encode_metadata, axis=1).values)
    
    # Create dataset from tensors
    image_paths_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    metadata_ds = tf.data.Dataset.from_tensor_slices(metadata_encoded)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    
    # Zip them together
    dataset = tf.data.Dataset.zip((
        (image_paths_ds, metadata_ds),
        labels_ds
    ))
    
    # Map with py_function for safe image loading
    def load_with_py_function(inputs, label):
        image_path, metadata = inputs
        
        # Use py_function to safely load image
        image, metadata = tf.py_function(
            func=lambda ip, m: (
                load_image(ip.numpy().decode('utf-8')),
                m
            ),
            inp=(image_path, metadata),
            Tout=(tf.float32, tf.float32)
        )
        
        # Set shapes
        image.set_shape((224, 224, 3))
        metadata.set_shape((metadata_encoded.shape[1],))
        
        # Apply augmentation if training
        if training:
            image = data_augmentation(image, training=True)
        
        return (image, metadata), label
    
    # Map with num_parallel_calls=1 for py_function safety
    dataset = dataset.map(load_with_py_function, num_parallel_calls=1)
    
    # Shuffle if requested
    if shuffle and training:
        dataset = dataset.shuffle(buffer_size=len(df))
    
    # Batch and prefetch
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    
    return dataset

# Get datasets
def get_datasets():
    """Load train and validation datasets"""
    train_ds = create_tf_dataset(
        SPLIT_DIR / "train.csv",
        training=True,
        shuffle=True
    )
    
    val_ds = create_tf_dataset(
        SPLIT_DIR / "val.csv",
        training=False,
        shuffle=False
    )
    
    return train_ds, val_ds

# K-fold datasets
def get_kfold_datasets(fold=0):
    """Load K-fold split"""
    kfold_dir = SPLIT_DIR / "kfold"
    
    train_ds = create_tf_dataset(
        kfold_dir / f"fold_{fold}_train.csv",
        training=True,
        shuffle=True
    )
    
    val_ds = create_tf_dataset(
        kfold_dir / f"fold_{fold}_val.csv",
        training=False,
        shuffle=False
    )
    
    return train_ds, val_ds

# Test
if __name__ == "__main__":
    print("Testing fixed data loader...")
    print("="*80)
    
    try:
        train_ds, val_ds = get_datasets()
        print("✅ TensorFlow datasets ready:")
        print(f"   Train batches: {len(train_ds)}")
        print(f"   Val batches: {len(val_ds)}")

        print("\n🧪 Sample batch inspection:")
        for (img, meta), label in train_ds.take(1):
            print(f"   Image shape: {img.shape}")
            print(f"   Image range: [{img.numpy().min():.2f}, {img.numpy().max():.2f}]")
            print(f"   Metadata shape: {meta.shape}")
            print(f"   Label shape: {label.shape}")
            
            if img.numpy().min() < -2 or img.numpy().max() > 2:
                print("\n   ⚠️ WARNING: Image values outside expected range!")
            else:
                print("\n   ✅ Image preprocessing looks correct!")
        
        print("\n" + "="*80)
        print("✅ Data loader test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()