import os
import re
import pandas as pd
from pathlib import Path

# ----------------------------
# CONFIGURATION
# ----------------------------
DATA_DIR = Path("data/raw")           # raw image folder
OUTPUT_DIR = Path("metadata")         # where CSV will be saved
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "dataset_metadata.csv"

# ----------------------------
# REGEX PATTERN to parse filenames
# ----------------------------
pattern = re.compile(
    r'^(?:([\d]+)-)?(Ashdown|Blood|MacConkey)_Agar_([A-Za-z]+)_([0-9]+)HR',
    re.IGNORECASE
)

# ----------------------------
# PARSE FILES
# ----------------------------
rows = []
for img_path in DATA_DIR.glob("*.png"):
    name = img_path.stem
    match = pattern.match(name)

    if match:
        sample_id, agar, species, time_hr = match.groups()
        sample_id = int(sample_id) if sample_id else None
        agar = agar.capitalize()
        species = species
        time_hr = int(time_hr)
    else:
        # fallback parsing for unexpected formats
        sample_id, agar, species, time_hr = None, None, None, None

    # Binary label: 1 if B. pseudomallei detected
    label = 1 if species and "pseudomallei" in species.lower() else 0

    rows.append({
        "filename": img_path.name,
        "filepath": str(img_path),
        "sample_id": sample_id,
        "agar": agar,
        "species": species,
        "time_hr": time_hr,
        "label_bpseudomallei": label
    })

# ----------------------------
# CREATE DATAFRAME
# ----------------------------
df = pd.DataFrame(rows)
df.to_csv(CSV_PATH, index=False)
print(f"✅ Metadata file created at: {CSV_PATH}")
print(f"Total images processed: {len(df)}")
print(df.head(10))

# ----------------------------
# OPTIONAL: Summary
# ----------------------------
summary = {
    "total_images": len(df),
    "bpseudomallei_count": int(df['label_bpseudomallei'].sum()),
    "non_bpseudomallei_count": int((1 - df['label_bpseudomallei']).sum()),
    "by_agar": df['agar'].value_counts(dropna=False).to_dict(),
    "by_species": df['species'].value_counts(dropna=False).to_dict()
}
print("\n📊 Summary:")
for k, v in summary.items():
    print(f"  {k}: {v}")
