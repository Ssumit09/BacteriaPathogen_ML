"""
fine_tune_improved.py

Fine-tunes the previously trained model (from train_improved.py),
unfreezes the top layers of the EfficientNet backbone, and fine-tunes it
with a smaller learning rate for better performance.

COMPATIBLE WITH:
- split_data_correctly.py (fixed splits + OneHotEncoder + StandardScaler)
- data_loader_fixed.py (multi-input with metadata encoding)
- train_improved.py (initial training)
"""

import tensorflow as tf
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import warnings

# Import FIXED data loader
from data_loader_fixed import get_datasets

warnings.filterwarnings('ignore')

# ----------------------------
# CONFIG
# ----------------------------
BASE_DIR = Path(".")
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
SPLITS_DIR = Path("metadata/splits_fixed")  # ✅ Use fixed splits

LOGS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = SPLITS_DIR / "train.csv"
VAL_CSV = SPLITS_DIR / "val.csv"

EPOCHS_FINE_TUNE = 50  # ✅ More epochs for fine-tuning
LEARNING_RATE_FINE_TUNE = 1e-5  # Much smaller LR for fine-tuning
BATCH_SIZE = 16

# ----------------------------
# 1️⃣ Load datasets (from fixed splits)
# ----------------------------
print("="*80)
print("LOADING FIXED DATASETS FOR FINE-TUNING")
print("="*80)
print("📂 Loading datasets from fixed splits...")

train_ds, val_ds = get_datasets()

# Detect metadata vector size
for (img_batch, meta_batch), label_batch in train_ds.take(1):
    METADATA_VECTOR_SIZE = int(meta_batch.shape[-1])
    print(f"✅ Detected metadata vector size: {METADATA_VECTOR_SIZE}")
    break

# ----------------------------
# 2️⃣ Load best model from initial training
# ----------------------------
print("\n" + "="*80)
print("LOADING PRE-TRAINED MODEL")
print("="*80)

best_model_path = MODELS_DIR / "best_model_improved.keras"

if not best_model_path.exists():
    raise FileNotFoundError(
        f"Best model checkpoint not found at {best_model_path}\n"
        "Please run train_improved.py first to train the initial model."
    )

print(f"📂 Loading model from {best_model_path}")
model = tf.keras.models.load_model(str(best_model_path), compile=False)
print("✅ Model loaded successfully!")

# Print current model summary
print("\n📊 Model Architecture:")
model.summary()

# ----------------------------
# 3️⃣ Unfreeze top layers of EfficientNet for fine-tuning
# ----------------------------
print("\n" + "="*80)
print("UNFREEZING TOP LAYERS FOR FINE-TUNING")
print("="*80)

# Get the EfficientNet base model
# The base model layers have names like "efficientnetb0_...", "block_...", etc.
base_layers = []
for i, layer in enumerate(model.layers):
    # Look for EfficientNet layers (they contain 'efficientnet' or 'block' in name)
    if 'efficientnetb0' in layer.name.lower() or 'block' in layer.name.lower():
        base_layers.append((i, layer))

if not base_layers:
    print("⚠️ Warning: EfficientNet layers not found by standard naming.")
    print("   Attempting alternative search...")
    # Fall back: find layers that are likely part of the backbone
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'trainable') and layer != model.layers[-1]:  # Not output layer
            base_layers.append((i, layer))

print(f"Found {len(base_layers)} potential base model layers")

if base_layers:
    start_idx = base_layers[0][0]
    end_idx = base_layers[-1][0]
    print(f"Base model layers span from index {start_idx} to {end_idx}")
else:
    raise ValueError("Could not find EfficientNet layers in model. Check model architecture.")

# Unfreeze top N% of EfficientNet layers for fine-tuning
unfreeze_percentage = 0.4  # Unfreeze top 40% of layers
num_layers_to_unfreeze = max(1, int(len(base_layers) * unfreeze_percentage))

print(f"\n🔓 Unfreezing top {unfreeze_percentage*100:.0f}% ({num_layers_to_unfreeze} layers)...")

# First, freeze ALL layers
for i, layer in enumerate(model.layers):
    layer.trainable = False

# Then unfreeze only the top layers
for idx, layer in base_layers[-num_layers_to_unfreeze:]:
    # Don't unfreeze batch norm layers (they should stay frozen)
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True
        print(f"   ✅ Unfroze: {layer.name}")

# Always allow metadata branch and head to be trainable
metadata_trainable = False
for layer in model.layers:
    if 'metadata' in layer.name.lower() or 'concat' in layer.name.lower() or 'output' in layer.name.lower():
        layer.trainable = True
        metadata_trainable = True
        print(f"   ✅ Unfroze: {layer.name}")

# Count trainable parameters
trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
total_count = model.count_params()
frozen_count = total_count - trainable_count

print(f"\n📊 Layer Status:")
print(f"   Total parameters: {total_count:,}")
print(f"   Trainable parameters: {trainable_count:,}")
print(f"   Frozen parameters: {frozen_count:,}")
print(f"   Trainable percentage: {100*trainable_count/total_count:.2f}%")

# ----------------------------
# 4️⃣ Recompile with smaller learning rate
# ----------------------------
print("\n" + "="*80)
print("RECOMPILING MODEL")
print("="*80)

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

print(f"✅ Model recompiled with learning rate: {LEARNING_RATE_FINE_TUNE}")

# ----------------------------
# 5️⃣ Compute class weights
# ----------------------------
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

# ----------------------------
# 6️⃣ Callbacks for fine-tuning
# ----------------------------
print("\n" + "="*80)
print("CONFIGURING CALLBACKS")
print("="*80)

callbacks = [
    ModelCheckpoint(
        str(MODELS_DIR / "best_model_finetuned.keras"),
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_auc',
        patience=10,  # ✅ Early stopping patience
        mode='max',
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_auc',
        factor=0.5,
        patience=3,
        mode='max',
        min_lr=1e-7,
        verbose=1
    ),
    CSVLogger(str(LOGS_DIR / "fine_tune_log.csv"))
]

print("✅ Callbacks configured:")
print(f"   - ModelCheckpoint: Saves best model based on val_auc")
print(f"   - EarlyStopping: Patience=10 epochs")
print(f"   - ReduceLROnPlateau: Reduces LR when plateaus")
print(f"   - CSVLogger: Logs to {LOGS_DIR / 'fine_tune_log.csv'}")

# ----------------------------
# 7️⃣ Fine-tune the model
# ----------------------------
print("\n" + "="*80)
print("STARTING FINE-TUNING")
print("="*80)
print(f"Epochs: {EPOCHS_FINE_TUNE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE_FINE_TUNE}")
print(f"Using fixed splits (no data leakage)")
print("="*80 + "\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE_TUNE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ----------------------------
# 8️⃣ Evaluate fine-tuned model
# ----------------------------
print("\n" + "="*80)
print("FINAL EVALUATION")
print("="*80)

results = model.evaluate(val_ds, verbose=0)
print("\nValidation Results (Fine-tuned Model):")
print(f"   Loss: {results[0]:.4f}")
print(f"   Accuracy: {results[1]:.4f}")
print(f"   Precision: {results[2]:.4f}")
print(f"   Recall: {results[3]:.4f}")
print(f"   AUC: {results[4]:.4f}")

# ----------------------------
# 9️⃣ Save final fine-tuned model
# ----------------------------
final_finetuned_path = MODELS_DIR / "final_finetuned_model.keras"
model.save(str(final_finetuned_path), include_optimizer=False)
print(f"\n✅ Final fine-tuned model saved to: {final_finetuned_path}")

# ----------------------------
# 🔟 Training Summary
# ----------------------------
print("\n" + "="*80)
print("FINE-TUNING SUMMARY")
print("="*80)

# Find best epoch
best_epoch = np.argmax(history.history['val_auc']) + 1
best_val_auc = max(history.history['val_auc'])
best_val_acc = history.history['val_accuracy'][best_epoch - 1]

print(f"\nBest Epoch: {best_epoch}")
print(f"   Val AUC: {best_val_auc:.4f}")
print(f"   Val Accuracy: {best_val_acc:.4f}")

# Compare with initial training (if logs exist)
initial_log_path = LOGS_DIR / "training_log_improved.csv"
if initial_log_path.exists():
    initial_log = pd.read_csv(initial_log_path)
    initial_best_auc = initial_log['val_auc'].max()
    improvement = (best_val_auc - initial_best_auc) / initial_best_auc * 100
    
    print(f"\nComparison with Initial Training:")
    print(f"   Initial best AUC: {initial_best_auc:.4f}")
    print(f"   Fine-tuned best AUC: {best_val_auc:.4f}")
    print(f"   Improvement: {improvement:+.2f}%")
    
    if improvement > 0:
        print("   ✅ Fine-tuning improved the model!")
    else:
        print("   ⚠️ Fine-tuning did not improve the model")
        print("      (This is normal if initial training already converged well)")

# Check for overfitting
train_acc = history.history['accuracy'][best_epoch - 1]
val_acc = best_val_acc
gap = train_acc - val_acc

print(f"\nOverfitting Check:")
print(f"   Train Accuracy: {train_acc:.4f}")
print(f"   Val Accuracy: {val_acc:.4f}")
print(f"   Gap: {gap:.4f}")

if gap > 0.15:
    print("   ⚠️ Significant overfitting detected")
elif gap > 0.08:
    print("   ⚠️ Moderate overfitting")
else:
    print("   ✅ Overfitting is under control")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("\n1. Generate classification reports:")
print("   python classification_report.py")
print("   python classification_report_by_agar.py")
print("\n2. Test on test set (if available):")
print("   python test_model.py")
print("\n3. Deploy the best model:")
print("   cp models/final_finetuned_model.keras backend/models/model.keras")

print("\n" + "="*80)
print("✅ FINE-TUNING COMPLETE!")
print("="*80)