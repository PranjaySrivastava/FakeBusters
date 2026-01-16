import os
import shutil
import random
import zipfile

print("Organizing CIFAKE dataset...")

# Extract ZIP if needed
zip_file = "cifake-real-and-ai-generated-synthetic-images.zip"
if os.path.exists(zip_file):
    print("Extracting ZIP...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall("data/cifake_raw")
    print("✓ Extracted")

# Find the folders - adjust these paths based on your extraction
# Check what was extracted
if os.path.exists("data/cifake_raw"):
    print("\nContents of extraction:")
    for item in os.listdir("data/cifake_raw"):
        print(f"  - {item}")

# Common paths after extraction
possible_paths = [
    ("data/cifake_raw/REAL", "data/cifake_raw/FAKE"),
    ("data/cifake_raw/real", "data/cifake_raw/fake"),
    ("data/cifake_raw/train/REAL", "data/cifake_raw/train/FAKE"),
]

real_source = None
fake_source = None

for real_path, fake_path in possible_paths:
    if os.path.exists(real_path) and os.path.exists(fake_path):
        real_source = real_path
        fake_source = fake_path
        break

if not real_source:
    print("\n❌ Could not find REAL and FAKE folders")
    print("Please manually check the extracted folders and update paths")
    exit(1)

print(f"\n✓ Found data:")
print(f"  Real: {real_source}")
print(f"  Fake: {fake_source}")

# Get files
real_files = [f for f in os.listdir(real_source) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
fake_files = [f for f in os.listdir(fake_source) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"\nFound {len(real_files):,} real images")
print(f"Found {len(fake_files):,} fake images")

# Use subset for faster training
TRAIN_SIZE = 5000  # Reduced for faster training
VAL_SIZE = 1000

random.seed(42)
random.shuffle(real_files)
random.shuffle(fake_files)

# Create folders
os.makedirs("data/cifake_organized/train/real", exist_ok=True)
os.makedirs("data/cifake_organized/train/fake", exist_ok=True)
os.makedirs("data/cifake_organized/val/real", exist_ok=True)
os.makedirs("data/cifake_organized/val/fake", exist_ok=True)

# Copy files
print("\nCopying files...")
for i, f in enumerate(real_files[:TRAIN_SIZE]):
    shutil.copy(os.path.join(real_source, f), f"data/cifake_organized/train/real/{f}")
    if (i+1) % 1000 == 0:
        print(f"  Real train: {i+1}/{TRAIN_SIZE}")

for i, f in enumerate(real_files[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]):
    shutil.copy(os.path.join(real_source, f), f"data/cifake_organized/val/real/{f}")

for i, f in enumerate(fake_files[:TRAIN_SIZE]):
    shutil.copy(os.path.join(fake_source, f), f"data/cifake_organized/train/fake/{f}")
    if (i+1) % 1000 == 0:
        print(f"  Fake train: {i+1}/{TRAIN_SIZE}")

for i, f in enumerate(fake_files[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]):
    shutil.copy(os.path.join(fake_source, f), f"data/cifake_organized/val/fake/{f}")

print("\n✓ Dataset organized!")
print(f"Train: {TRAIN_SIZE} per class")
print(f"Val: {VAL_SIZE} per class")