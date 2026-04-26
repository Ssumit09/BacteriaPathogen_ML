import cv2
import os

# Input and output folders
input_folder = "data/OriginalImages"   # change this
output_folder = "data/raw" # change this

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in input folder
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)
    
    # Read image
    img = cv2.imread(file_path)
    
    # Skip if not an image
    if img is None:
        continue
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Save image
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, gray)

print("✅ All images converted to grayscale successfully!")