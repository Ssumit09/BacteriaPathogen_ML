"""
train_improved.py

Improved training script with:
1. Uses fixed data splits (no leakage)
2. More conservative fine-tuning (freeze more layers)
3. Better callbacks and monitoring
4. L2 regularization
"""

import os
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras import regularizers

# Import FIXED data loader
from data_loader_fixed import get_datasets

# Import model builder
from model import build_multiinput_model

# ===========================
# CONFIG
# ===========================
BASE_DIR = Path(".")
METADATA_DIR = BASE_DIR / "metadata"
SPLITS_DIR = METADATA_DIR / "splits_fixed"  # ✅ Use fixed splits
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = SPLITS_DIR / "train.csv"
VAL_CSV = SPLITS_DIR / "val.csv"

IMG_SHAPE = (224, 224, 3)
METADATA_VECTOR_SIZE = None  # Auto-detect
BATCH_SIZE = 16
EPOCHS = 100  # ✅ More epochs with early stopping

# ===========================
# 1️⃣ Load Datasets
# ===========================
print("="*80)
print("LOADING FIXED DATASETS")
print("="*80)
print("📂 Creating tf.data datasets from FIXED splits...")
print(f"   Train CSV: {TRAIN_CSV}")
print(f"   Val CSV: {VAL_CSV}")

train_ds, val_ds = get_datasets()

# Detect metadata vector size
for (img_batch, meta_batch), label_batch in train_ds.take(1):
    METADATA_VECTOR_SIZE = int(meta_batch.shape[-1])
    print(f"\n✅ Detected metadata vector size: {METADATA_VECTOR_SIZE}")
    print(f"   Image batch shape: {img_batch.shape}")
    print(f"   Image value range: [{img_batch.numpy().min():.2f}, {img_batch.numpy().max():.2f}]")
    
    # Verify preprocessing
    if img_batch.numpy().min() < -2 or img_batch.numpy().max() > 2:
        print("\n⚠️ WARNING: Unexpected image value range!")
        print("   Expected: ~[-1, 1] (EfficientNet preprocessing)")
    else:
        print("✅ Image preprocessing looks correct!")

# ===========================
# 2️⃣ Build Model with More Regularization
# ===========================
print("\n" + "="*80)
print("BUILDING MODEL")
print("="*80)

model = build_multiinput_model(
    input_image_shape=IMG_SHAPE,
    metadata_vector_size=METADATA_VECTOR_SIZE,
    base_trainable=False,      # ✅ Freeze EfficientNet initially
    dropout_rate=0.5,           # ✅ Increased from 0.3
    learning_rate=1e-4
)

print("\n📊 Model Summary:")
model.summary()

total_params = model.count_params()
trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
frozen_params = total_params - trainable_params

print(f"\n📈 Model Statistics:")
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"   Frozen parameters: {frozen_params:,}")

# ===========================
# 3️⃣ Compute Class Weights
# ===========================
print("\n" + "="*80)
print("COMPUTING CLASS WEIGHTS")
print("="*80)

train_df = pd.read_csv(TRAIN_CSV)
y_train = train_df['label_bpseudomallei'].values

print(f"Training set class distribution:")
print(f"   Class 0 (Other): {(y_train == 0).sum()} samples")
print(f"   Class 1 (B. pseudomallei): {(y_train == 1).sum()} samples")

classes = np.unique(y_train)
cw = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = {int(classes[i]): float(cw[i]) for i in range(len(classes))}

print(f"\n⚖️ Class weights: {class_weights}")

# ===========================
# 4️⃣ Callbacks
# ===========================
print("\n" + "="*80)
print("CONFIGURING CALLBACKS")
print("="*80)

checkpoint_path = MODELS_DIR / "best_model_improved.keras"
callbacks = [
    ModelCheckpoint(
        str(checkpoint_path), 
        monitor='val_auc', 
        mode='max', 
        save_best_only=True, 
        verbose=1,
        save_weights_only=False
    ),
    EarlyStopping(
        monitor='val_auc', 
        patience=15,  # ✅ Increased patience
        mode='max', 
        restore_best_weights=True, 
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_auc', 
        factor=0.5, 
        patience=5,  # ✅ Increased patience
        mode='max', 
        min_lr=1e-7,
        verbose=1
    ),
    CSVLogger(str(LOGS_DIR / "training_log_improved.csv"))
]

print("✅ Callbacks configured:")
print(f"   - ModelCheckpoint: Saves best model based on val_auc")
print(f"   - EarlyStopping: Patience=15 epochs")
print(f"   - ReduceLROnPlateau: Reduces LR when plateaus")
print(f"   - CSVLogger: Logs to {LOGS_DIR / 'training_log_improved.csv'}")

# ===========================
# 5️⃣ Train
# ===========================
print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Initial learning rate: 1e-4")
print(f"Using fixed splits (no data leakage)")
print("="*80 + "\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# ===========================
# 6️⃣ Evaluate
# ===========================
print("\n" + "="*80)
print("FINAL EVALUATION")
print("="*80)

results = model.evaluate(val_ds, verbose=0)
print("\nValidation Results:")
print(f"   Loss: {results[0]:.4f}")
print(f"   Accuracy: {results[1]:.4f}")
print(f"   Precision: {results[2]:.4f}")
print(f"   Recall: {results[3]:.4f}")
print(f"   AUC: {results[4]:.4f}")

# ===========================
# 7️⃣ Save Final Model
# ===========================
final_model_path = MODELS_DIR / "final_model_improved.keras"
model.save(str(final_model_path), include_optimizer=False)
print(f"\n✅ Final model saved to: {final_model_path}")

# ===========================
# 8️⃣ Training Summary
# ===========================
print("\n" + "="*80)
print("TRAINING SUMMARY")
print("="*80)

# Find best epoch
best_epoch = np.argmax(history.history['val_auc']) + 1
best_val_auc = max(history.history['val_auc'])
best_val_acc = history.history['val_accuracy'][best_epoch - 1]

print(f"\nBest Epoch: {best_epoch}")
print(f"   Val AUC: {best_val_auc:.4f}")
print(f"   Val Accuracy: {best_val_acc:.4f}")

# Check for overfitting
train_acc = history.history['accuracy'][best_epoch - 1]
val_acc = best_val_acc
gap = train_acc - val_acc

print(f"\nOverfitting Check:")
print(f"   Train Accuracy: {train_acc:.4f}")
print(f"   Val Accuracy: {val_acc:.4f}")
print(f"   Gap: {gap:.4f}")

if gap > 0.15:
    print("\n⚠️ WARNING: Significant overfitting detected!")
    print("   Recommendations:")
    print("   1. Increase dropout (currently 0.5)")
    print("   2. Add more augmentation")
    print("   3. Collect more training data")
    print("   4. Use a smaller model")
elif gap > 0.08:
    print("\n⚠️ Moderate overfitting detected")
    print("   Model may benefit from more regularization")
else:
    print("\n✅ Overfitting is under control")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("\n1. Run fine-tuning (optional):")
print("   python fine_tune_improved.py")
print("\n2. Generate classification reports:")
print("   python classification_report.py")
print("   python classification_report_by_agar.py")
print("\n3. Update backend API:")
print("   Replace backend/app.py with backend/app_fixed.py")
print("\n4. Test with React Native app")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)