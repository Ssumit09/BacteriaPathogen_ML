# """
# fine_tune.py

# This script loads the previously trained model (from src/train.py),
# unfreezes the top layers of the EfficientNet backbone, and fine-tunes it
# with a smaller learning rate for better performance.
# """

# import tensorflow as tf
# from pathlib import Path
# import pandas as pd
# import numpy as np
# from sklearn.utils import class_weight
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

# from data_loader import get_datasets
# from model import build_multiinput_model

# # ----------------------------
# # CONFIG
# # ----------------------------
# BASE_DIR = Path(".")
# MODELS_DIR = BASE_DIR / "models"
# LOGS_DIR = BASE_DIR / "logs"
# SPLITS_DIR = Path("metadata/splits")

# LOGS_DIR.mkdir(parents=True, exist_ok=True)

# TRAIN_CSV = SPLITS_DIR / "train.csv"
# VAL_CSV = SPLITS_DIR / "val.csv"
# TEST_CSV = SPLITS_DIR / "test.csv"

# EPOCHS_FINE_TUNE = 20
# LEARNING_RATE_FINE_TUNE = 1e-5   # much smaller LR for fine-tuning

# # ----------------------------
# # 1️⃣ Load datasets (same as before)
# # ----------------------------
# train_ds, val_ds, test_ds = get_datasets()

# # Detect metadata vector size
# for (img_batch, meta_batch), label_batch in train_ds.take(1):
#     METADATA_VECTOR_SIZE = int(meta_batch.shape[-1])
#     break

# # ----------------------------
# # 2️⃣ Load best model from previous training
# # ----------------------------
# best_model_path = MODELS_DIR / "best_model.keras"
# if not best_model_path.exists():
#     raise FileNotFoundError("Best model checkpoint not found. Train the base model first.")

# print(f"📂 Loading model from {best_model_path}")
# model = tf.keras.models.load_model(best_model_path, compile=False)

# # ----------------------------
# # 3️⃣ Unfreeze top layers of EfficientNet for fine-tuning
# # ----------------------------
# print("\n🔓 Unfreezing top layers for fine-tuning...")

# # Find the first and last EfficientNet layer indices
# eff_layers = [i for i, layer in enumerate(model.layers) if layer.name.startswith("block")]
# if not eff_layers:
#     raise ValueError("EfficientNet layers not found in loaded model.")

# start_idx = min(eff_layers)
# end_idx = max(eff_layers)
# efficientnet_layers = model.layers[start_idx:end_idx + 1]

# print(f"Found EfficientNet layers from index {start_idx} to {end_idx}.")
# print(f"Total EfficientNet layers detected: {len(efficientnet_layers)}")

# # Unfreeze top N layers (for example, last 40)
# N = 40
# for layer in efficientnet_layers[-N:]:
#     if not isinstance(layer, tf.keras.layers.BatchNormalization):
#         layer.trainable = True

# for layer in efficientnet_layers[:-N]:
#     layer.trainable = False

# trainable_count = np.sum([layer.trainable for layer in efficientnet_layers])
# print(f"✅ Fine-tuning {trainable_count} EfficientNet layers.")
# # # ----------------------------
# # # 3️⃣ Unfreeze top layers of EfficientNet for fine-tuning
# # # ----------------------------
# # print("\n🔓 Unfreezing top layers for fine-tuning...")

# # # Find EfficientNet base
# # base_model = None
# # for layer in model.layers:
# #     if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
# #         base_model = layer
# #         break

# # if base_model is None:
# #     raise ValueError("EfficientNet base model not found in loaded model.")

# # # Unfreeze top layers (e.g., last 40)
# # for layer in base_model.layers[-40:]:
# #     if not isinstance(layer, tf.keras.layers.BatchNormalization):
# #         layer.trainable = True

# # # Keep lower layers frozen
# # for layer in base_model.layers[:-40]:
# #     layer.trainable = False

# # print(f"Total layers in EfficientNet: {len(base_model.layers)}")
# # trainable_count = np.sum([layer.trainable for layer in base_model.layers])
# # print(f"✅ Fine-tuning {trainable_count} top layers.")

# # ----------------------------
# # 4️⃣ Recompile with smaller learning rate
# # ----------------------------
# optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE_TUNE)
# model.compile(
#     optimizer=optimizer,
#     loss='binary_crossentropy',
#     metrics=[
#         'accuracy',
#         tf.keras.metrics.Precision(name='precision'),
#         tf.keras.metrics.Recall(name='recall'),
#         tf.keras.metrics.AUC(name='auc')
#     ]
# )

# # ----------------------------
# # 5️⃣ Compute class weights (again)
# # ----------------------------
# train_df = pd.read_csv(TRAIN_CSV)
# y_train = train_df['label_bpseudomallei'].values
# classes = np.unique(y_train)
# cw = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
# class_weights = {int(classes[i]): float(cw[i]) for i in range(len(classes))}
# print("Class weights:", class_weights)

# # ----------------------------
# # 6️⃣ Callbacks
# # ----------------------------
# # callbacks = [
# #     ModelCheckpoint(str(MODELS_DIR / "best_model_finetuned.h5"), monitor='val_auc', mode='max', save_best_only=True, verbose=1),
# #     EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True, verbose=1),
# #     ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=2, mode='max', verbose=1),
# #     CSVLogger(LOGS_DIR / "fine_tune_log.csv")
# # ]
# callbacks = [
#     ModelCheckpoint(str(MODELS_DIR / "best_model_finetuned.keras"), monitor='val_auc', mode='max', save_best_only=True, verbose=1),
#     EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True, verbose=1),
#     ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=2, mode='max', verbose=1)
# ]


# # ----------------------------
# # 7️⃣ Train (fine-tune)
# # ----------------------------
# print("\n🚀 Starting fine-tuning...")
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=EPOCHS_FINE_TUNE,
#     class_weight=class_weights,
#     callbacks=callbacks
# )

# # ----------------------------
# # 8️⃣ Evaluate final model
# # ----------------------------
# print("\n📊 Evaluating fine-tuned model on test set...")
# results = model.evaluate(test_ds)
# print("Test results (loss, accuracy, precision, recall, auc):", results)

# # Save final fine-tuned model
# final_finetuned_path = MODELS_DIR / "final_finetuned_model.keras"
# model.save(final_finetuned_path, include_optimizer=False)
# print(f"\n✅ Fine-tuned model saved to: {final_finetuned_path}")



# --------------------------------------
"""
fine_tune.py

This script loads the previously trained model (from src/train.py),
unfreezes the top layers of the EfficientNet backbone, and fine-tunes it
with a smaller learning rate for better performance.
"""

import tensorflow as tf
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

from data_loader import get_datasets

# ----------------------------
# CONFIG
# ----------------------------
BASE_DIR = Path(".")
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
SPLITS_DIR = Path("metadata/splits")

LOGS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = SPLITS_DIR / "train.csv"
VAL_CSV = SPLITS_DIR / "val.csv"

EPOCHS_FINE_TUNE = 20
LEARNING_RATE_FINE_TUNE = 1e-5   # much smaller LR for fine-tuning

# ----------------------------
# 1️⃣ Load datasets (80-20 split)
# ----------------------------
train_ds, val_ds = get_datasets()

# Detect metadata vector size
for (img_batch, meta_batch), label_batch in train_ds.take(1):
    METADATA_VECTOR_SIZE = int(meta_batch.shape[-1])
    break

# ----------------------------
# 2️⃣ Load best model from previous training
# ----------------------------
best_model_path = MODELS_DIR / "best_model.keras"
if not best_model_path.exists():
    raise FileNotFoundError("Best model checkpoint not found. Train the base model first.")

print(f"📂 Loading model from {best_model_path}")
model = tf.keras.models.load_model(str(best_model_path), compile=False)

# ----------------------------
# 3️⃣ Unfreeze top layers of EfficientNet for fine-tuning
# ----------------------------
print("\n🔓 Unfreezing top layers for fine-tuning...")

# Find the first and last EfficientNet layer indices
eff_layers = [i for i, layer in enumerate(model.layers) if layer.name.startswith("block")]
if not eff_layers:
    raise ValueError("EfficientNet layers not found in loaded model.")

start_idx = min(eff_layers)
end_idx = max(eff_layers)
efficientnet_layers = model.layers[start_idx:end_idx + 1]

print(f"Found EfficientNet layers from index {start_idx} to {end_idx}.")
print(f"Total EfficientNet layers detected: {len(efficientnet_layers)}")

# Unfreeze top N layers (for example, last 40)
N = 40
for layer in efficientnet_layers[-N:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

for layer in efficientnet_layers[:-N]:
    layer.trainable = False

trainable_count = np.sum([layer.trainable for layer in efficientnet_layers])
print(f"✅ Fine-tuning {trainable_count} EfficientNet layers.")

# ----------------------------
# 4️⃣ Recompile with smaller learning rate
# ----------------------------
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE_TUNE)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

# ----------------------------
# 5️⃣ Compute class weights
# ----------------------------
train_df = pd.read_csv(TRAIN_CSV)
y_train = train_df['label_bpseudomallei'].values
classes = np.unique(y_train)
cw = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = {int(classes[i]): float(cw[i]) for i in range(len(classes))}
print("\n⚖️ Class weights:", class_weights)

# ----------------------------
# 6️⃣ Callbacks
# ----------------------------
callbacks = [
    ModelCheckpoint(str(MODELS_DIR / "best_model_finetuned.keras"), monitor='val_auc', mode='max', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=2, mode='max', verbose=1),
    CSVLogger(str(LOGS_DIR / "fine_tune_log.csv"))
]

# ----------------------------
# 7️⃣ Train (fine-tune)
# ----------------------------
print("\n🚀 Starting fine-tuning...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE_TUNE,
    class_weight=class_weights,
    callbacks=callbacks
)

# ----------------------------
# 8️⃣ Evaluate final model
# ----------------------------
print("\n📊 Evaluating fine-tuned model on validation set...")
results = model.evaluate(val_ds)
print("Validation results (loss, accuracy, precision, recall, auc):", results)

# Save final fine-tuned model
final_finetuned_path = MODELS_DIR / "final_finetuned_model.keras"
model.save(str(final_finetuned_path), include_optimizer=False)
print(f"\n✅ Fine-tuned model saved to: {final_finetuned_path}")