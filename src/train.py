"""
Train script for the multi-input model.

Expected project layout:
- metadata/splits/train.csv, val.csv, test.csv
- metadata/encoders/onehot_encoder.pkl and scaler.pkl
- src/data_loader.py exists and creates `train_ds`, `val_ds`, `test_ds`
"""

import os
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

# import model builder and dataset creation helper
from model import build_multiinput_model
from data_loader import create_tf_dataset  # assumes create_tf_dataset(csv_path) returns tf.data.Dataset

# CONFIG
BASE_DIR = Path(".")
METADATA_DIR = BASE_DIR / "metadata"
SPLITS_DIR = METADATA_DIR / "splits"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = SPLITS_DIR / "train.csv"
VAL_CSV = SPLITS_DIR / "val.csv"
TEST_CSV = SPLITS_DIR / "test.csv"

IMG_SHAPE = (224, 224, 3)
METADATA_VECTOR_SIZE = None  # Auto-detect below
BATCH_SIZE = 16
EPOCHS = 50

# 1) Create tf.data datasets
print("Creating tf.data datasets...")
train_ds = create_tf_dataset(TRAIN_CSV)
val_ds = create_tf_dataset(VAL_CSV)
test_ds = create_tf_dataset(TEST_CSV)

# detect metadata vector size by inspecting a batch
for (img_batch, meta_batch), label_batch in train_ds.take(1):
    METADATA_VECTOR_SIZE = int(meta_batch.shape[-1])
    print("Detected metadata vector size:", METADATA_VECTOR_SIZE)

# 2) Build model
print("Building model...")
model = build_multiinput_model(
    input_image_shape=IMG_SHAPE,
    metadata_vector_size=METADATA_VECTOR_SIZE,
    base_trainable=False,      # freeze base model initially
    dropout_rate=0.3,
    learning_rate=1e-4
)
model.summary()

# 3) Compute class weights (to handle imbalance)
train_df = pd.read_csv(TRAIN_CSV)
y_train = train_df['label_bpseudomallei'].values
classes = np.unique(y_train)
cw = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = {int(classes[i]): float(cw[i]) for i in range(len(classes))}
print("Class weights:", class_weights)

# 4) Callbacks
checkpoint_path = MODELS_DIR / "best_model.h5"
callbacks = [
    ModelCheckpoint(str(checkpoint_path), monitor='val_auc', mode='max', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_auc', patience=8, mode='max', restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, mode='max', verbose=1),
    CSVLogger(LOGS_DIR / "training_log.csv")
]

# 5) Train
steps_per_epoch = None  # let TF infer from dataset
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

# 6) Save final model (SavedModel + H5 checkpoint is already saved by ModelCheckpoint)
final_saved = MODELS_DIR / "final_saved_model.h5"
model.save(final_saved, include_optimizer=False)
print("Model saved to:", final_saved)

# 7) Evaluate on test set
print("Evaluating on test set...")
results = model.evaluate(test_ds)
print("Test results (loss, accuracy, precision, recall, auc):", results)
