import os
import zipfile
import subprocess

# Set your Kaggle dataset path (update this)
dataset = "username/dataset-name"  # e.g. "rohanrao/indian-soil-types"

# Define paths
raw_data_dir = "data/raw"
zip_path = os.path.join(raw_data_dir, "dataset.zip")

# Ensure the raw data directory exists
os.makedirs(raw_data_dir, exist_ok=True)

# Download from Kaggle using subprocess (requires kaggle.json config)
print("Downloading from Kaggle...")
result = subprocess.run([
    "kaggle", "datasets", "download", "-d", dataset, "-p", raw_data_dir
], capture_output=True, text=True)

if result.returncode != 0:
    print("❌ Download failed:", result.stderr)
    exit(1)

print("✅ Download complete.")

# Unzip all ZIP files in the raw directory
for file in os.listdir(raw_data_dir):
    if file.endswith(".zip"):
        print(f"Extracting {file}...")
        with zipfile.ZipFile(os.path.join(raw_data_dir, file), 'r') as zip_ref:
            zip_ref.extractall(raw_data_dir)

print("✅ Extraction complete.")
