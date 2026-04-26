"""
split_data_correctly.py

CRITICAL FIX: Split BEFORE augmentation to prevent data leakage
This ensures validation set contains truly unseen images
ADDED: OneHotEncoder for categorical + StandardScaler for numeric features
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import shutil
import numpy as np
import joblib

# --------------------------------
# CONFIGURATION
# --------------------------------
CSV_PATH = Path("metadata/dataset_metadata.csv")
OUTPUT_DIR = Path("metadata/splits_fixed")
RAW_DATA_DIR = Path("data/raw")
SPLIT_DATA_DIR = Path("data/split")
ENCODER_DIR = Path("metadata/encoders")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DATA_DIR.mkdir(parents=True, exist_ok=True)
ENCODER_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------
# LOAD ORIGINAL METADATA
# --------------------------------
print("📂 Loading dataset metadata...")
df = pd.read_csv(CSV_PATH)

print(f"\nTotal images in dataset: {len(df)}")
print(f"B. pseudomallei: {df['label_bpseudomallei'].sum()}")
print(f"Other bacteria: {(1 - df['label_bpseudomallei']).sum()}")

# --------------------------------
# IDENTIFY ORIGINAL (NON-AUGMENTED) IMAGES ONLY
# --------------------------------
print("\n🔍 Filtering to original images only (removing augmented versions)...")

# Identify augmented images by common prefixes
augmentation_prefixes = ['rot_', 'flip_', 'bright_', 'zoom_', 'aug_']

def is_original_image(filename):
    """Check if image is original (not augmented)"""
    filename_lower = filename.lower()
    for prefix in augmentation_prefixes:
        if filename_lower.startswith(prefix):
            return False
    return True

# Filter to original images only
df['is_original'] = df['filename'].apply(is_original_image)
original_df = df[df['is_original']].copy()

print(f"\n✅ Found {len(original_df)} original images (excluding augmented)")
print(f"   B. pseudomallei: {original_df['label_bpseudomallei'].sum()}")
print(f"   Other bacteria: {(1 - original_df['label_bpseudomallei']).sum()}")

# --------------------------------
# ENCODE METADATA (categorical + numerical)
# --------------------------------
print("\n🔤 Encoding metadata features...")

# Define categorical and numeric columns
categorical_cols = ['agar', 'species']
numeric_cols = ['time_hr']

# Fit OneHotEncoder on ORIGINAL DATA (before split)
print("   Fitting OneHotEncoder on original data...")
onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
onehot_encoder.fit(original_df[categorical_cols])
encoded_cats = onehot_encoder.transform(original_df[categorical_cols])
encoded_cat_columns = onehot_encoder.get_feature_names_out(categorical_cols)

# Fit StandardScaler on ORIGINAL DATA (before split)
print("   Fitting StandardScaler on original data...")
scaler = StandardScaler()
scaler.fit(original_df[numeric_cols])
scaled_nums = scaler.transform(original_df[numeric_cols])

# Combine encoded metadata
encoded_metadata = pd.concat([
    pd.DataFrame(encoded_cats, columns=encoded_cat_columns),
    pd.DataFrame(scaled_nums, columns=numeric_cols)
], axis=1)

# Create encoded dataframe by combining original with encoded features
original_df_encoded = pd.concat([
    original_df.reset_index(drop=True), 
    encoded_metadata.reset_index(drop=True)
], axis=1)

print(f"✅ Encoding completed")
print(f"   Categorical features: {len(encoded_cat_columns)}")
print(f"   Numeric features: {len(numeric_cols)}")

# --------------------------------
# SPLIT ORIGINAL IMAGES ONLY (80-20)
# --------------------------------
print("\n✂️ Splitting original images (80% train, 20% val)...")

train_df, val_df = train_test_split(
    original_df_encoded,
    test_size=0.2,
    stratify=original_df_encoded['label_bpseudomallei'],
    random_state=42
)

print(f"\n📊 Split Results:")
print(f"   Train: {len(train_df)} images")
print(f"     - B. pseudomallei: {train_df['label_bpseudomallei'].sum()}")
print(f"     - Other bacteria: {(1 - train_df['label_bpseudomallei']).sum()}")
print(f"\n   Val: {len(val_df)} images")
print(f"     - B. pseudomallei: {val_df['label_bpseudomallei'].sum()}")
print(f"     - Other bacteria: {(1 - val_df['label_bpseudomallei']).sum()}")

# --------------------------------
# SAVE SPLITS
# --------------------------------
train_csv = OUTPUT_DIR / "train.csv"
val_csv = OUTPUT_DIR / "val.csv"

train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)

print(f"\n✅ Splits saved:")
print(f"   {train_csv}")
print(f"   {val_csv}")

# --------------------------------
# SAVE ENCODERS FOR INFERENCE
# --------------------------------
print("\n💾 Saving encoders for inference...")

joblib.dump(onehot_encoder, ENCODER_DIR / "onehot_encoder.pkl")
joblib.dump(scaler, ENCODER_DIR / "scaler.pkl")

print(f"   Saved OneHotEncoder → {ENCODER_DIR / 'onehot_encoder.pkl'}")
print(f"   Saved StandardScaler → {ENCODER_DIR / 'scaler.pkl'}")

# --------------------------------
# PREVIEW ENCODED FEATURES
# --------------------------------
print("\n🧪 Encoded feature sample (first 5 rows):")
print(train_df.head()[encoded_cat_columns.tolist() + numeric_cols])

# --------------------------------
# COPY IMAGES TO SEPARATE FOLDERS
# --------------------------------
print("\n📁 Organizing images into train/val folders...")

train_img_dir = SPLIT_DATA_DIR / "train"
val_img_dir = SPLIT_DATA_DIR / "val"

train_img_dir.mkdir(parents=True, exist_ok=True)
val_img_dir.mkdir(parents=True, exist_ok=True)

# Copy training images
print("   Copying training images...")
for _, row in train_df.iterrows():
    src = RAW_DATA_DIR / row['filename']
    dst = train_img_dir / row['filename']
    if src.exists():
        shutil.copy2(src, dst)

# Copy validation images
print("   Copying validation images...")
for _, row in val_df.iterrows():
    src = RAW_DATA_DIR / row['filename']
    dst = val_img_dir / row['filename']
    if src.exists():
        shutil.copy2(src, dst)

print(f"\n✅ Images organized:")
print(f"   Train: {train_img_dir}")
print(f"   Val: {val_img_dir}")

# --------------------------------
# GENERATE K-FOLD SPLITS (Recommended for small datasets)
# --------------------------------
print("\n🔄 Generating 5-Fold Cross-Validation splits...")

kfold_dir = OUTPUT_DIR / "kfold"
kfold_dir.mkdir(parents=True, exist_ok=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y = original_df_encoded['label_bpseudomallei'].values

for fold, (train_idx, val_idx) in enumerate(skf.split(original_df_encoded, y)):
    fold_train_df = original_df_encoded.iloc[train_idx]
    fold_val_df = original_df_encoded.iloc[val_idx]
    
    fold_train_df.to_csv(kfold_dir / f"fold_{fold}_train.csv", index=False)
    fold_val_df.to_csv(kfold_dir / f"fold_{fold}_val.csv", index=False)
    
    print(f"   Fold {fold}: Train={len(fold_train_df)}, Val={len(fold_val_df)}")

print(f"\n✅ K-Fold splits saved in: {kfold_dir}")

# --------------------------------
# ANALYSIS: Check for Data Leakage
# --------------------------------
print("\n🔍 Checking for data leakage...")

train_files = set(train_df['filename'].values)
val_files = set(val_df['filename'].values)

overlap = train_files.intersection(val_files)

if len(overlap) == 0:
    print("   ✅ NO DATA LEAKAGE: Train and Val sets are completely independent")
else:
    print(f"   ⚠️ WARNING: {len(overlap)} files appear in both train and val!")
    print(f"   Overlapping files: {list(overlap)[:5]}...")

# --------------------------------
# PER-AGAR DISTRIBUTION
# --------------------------------
print("\n📊 Per-Agar Distribution:")
print("\nTraining Set:")
print(train_df.groupby('agar')['label_bpseudomallei'].agg(['count', 'sum', 'mean']))

print("\nValidation Set:")
print(val_df.groupby('agar')['label_bpseudomallei'].agg(['count', 'sum', 'mean']))

# --------------------------------
# WARNINGS AND RECOMMENDATIONS
# --------------------------------
print("\n" + "="*80)
print("⚠️ IMPORTANT NOTES:")
print("="*80)

total_original = len(original_df)
if total_original < 200:
    print(f"⚠️ Small dataset ({total_original} images)")
    print("   Recommendations:")
    print("   1. Use K-Fold cross-validation for better performance estimates")
    print("   2. Apply aggressive augmentation during training")
    print("   3. Use a smaller model or freeze more layers")
    print("   4. Consider collecting more data if possible")

# Check class balance
class_balance = original_df['label_bpseudomallei'].mean()
if class_balance < 0.3 or class_balance > 0.7:
    print(f"\n⚠️ Class imbalance detected ({class_balance:.2%} positive)")
    print("   Make sure to use class weights during training")

# Check per-agar samples
print("\n📋 Agar-wise sample counts:")
agar_counts = original_df.groupby('agar').size()
for agar, count in agar_counts.items():
    status = "⚠️ Very few samples" if count < 20 else "✅ Good"
    print(f"   {agar}: {count} images {status}")

print("\n" + "="*80)
print("✅ Data split and encoding completed correctly!")
print("="*80)
print("\nNext steps:")
print("1. Update data_loader.py to use 'metadata/splits_fixed/' instead of 'metadata/splits/'")
print("2. Load encoders during inference:")
print("   encoder = joblib.load('metadata/encoders/onehot_encoder.pkl')")
print("   scaler = joblib.load('metadata/encoders/scaler.pkl')")
print("3. Transform new data using loaded encoders")
print("4. Augmentation will now ONLY be applied to training set")
print("5. Validation results will reflect TRUE model performance")
print("\n💡 For best results with small dataset, use K-Fold cross-validation")