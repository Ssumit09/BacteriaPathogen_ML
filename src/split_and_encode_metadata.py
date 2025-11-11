import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

# --------------------------------
# CONFIGURATION
# --------------------------------
CSV_PATH = Path("metadata/dataset_metadata.csv")  # already generated file
SPLIT_DIR = Path("metadata/splits")
ENCODER_DIR = Path("metadata/encoders")

SPLIT_DIR.mkdir(parents=True, exist_ok=True)
ENCODER_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------
# LOAD METADATA CSV
# --------------------------------
print("📂 Loading dataset metadata from:", CSV_PATH)
df = pd.read_csv(CSV_PATH)

# Basic sanity check
if "label_bpseudomallei" not in df.columns:
    raise ValueError("CSV missing label column. Please ensure dataset_metadata.csv was generated correctly.")

# --------------------------------
# SPLIT DATASET (stratified to maintain class balance)
# --------------------------------
train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df['label_bpseudomallei'], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['label_bpseudomallei'], random_state=42
)

# Save splits
train_df.to_csv(SPLIT_DIR / "train.csv", index=False)
val_df.to_csv(SPLIT_DIR / "val.csv", index=False)
test_df.to_csv(SPLIT_DIR / "test.csv", index=False)

print(f"✅ Dataset split completed:")
print(f"   Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")

# --------------------------------
# ENCODE METADATA (categorical + numerical)
# --------------------------------
categorical_cols = ['agar', 'species']
numeric_cols = ['time_hr']

# One-hot encode categorical columns
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cats = encoder.fit_transform(df[categorical_cols])
encoded_cat_columns = encoder.get_feature_names_out(categorical_cols)

# Standardize numeric columns
scaler = StandardScaler()
scaled_nums = scaler.fit_transform(df[numeric_cols])

# Combine encoded metadata
encoded_metadata = pd.concat([
    pd.DataFrame(encoded_cats, columns=encoded_cat_columns),
    pd.DataFrame(scaled_nums, columns=numeric_cols)
], axis=1)

# Merge encoded metadata back to dataframe (optional for inspection)
encoded_df = pd.concat([df.reset_index(drop=True), encoded_metadata], axis=1)

# Save encoders for reuse during model inference
joblib.dump(encoder, ENCODER_DIR / "onehot_encoder.pkl")
joblib.dump(scaler, ENCODER_DIR / "scaler.pkl")

print("\n✅ Metadata encoding completed.")
print("   Saved OneHotEncoder →", ENCODER_DIR / "onehot_encoder.pkl")
print("   Saved Scaler →", ENCODER_DIR / "scaler.pkl")

# --------------------------------
# Optional: Preview encoded metadata
# --------------------------------
print("\n🧠 Encoded feature sample (first 5 rows):")
print(encoded_df.head()[encoded_cat_columns.tolist() + numeric_cols])
