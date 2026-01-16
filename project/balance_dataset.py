import os
import shutil
import random

print("Checking dataset balance...")

real_path = 'data/images/real'
fake_path = 'data/images/fake'

# Count images
real_files = [f for f in os.listdir(real_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
fake_files = [f for f in os.listdir(fake_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

real_count = len(real_files)
fake_count = len(fake_files)

print(f"\nCurrent dataset:")
print(f"  Real: {real_count:,}")
print(f"  Fake: {fake_count:,}")
print(f"  Imbalance ratio: {fake_count/real_count:.2f}:1 (fake:real)")

# Target: balance to smaller class
target_count = min(real_count, fake_count)

print(f"\n✂️  Balancing to {target_count:,} images per class...")

# Create balanced folders
os.makedirs('data/images_balanced/real', exist_ok=True)
os.makedirs('data/images_balanced/fake', exist_ok=True)

# Copy real images
print(f"\nCopying {target_count:,} real images...")
selected_real = random.sample(real_files, target_count) if real_count > target_count else real_files
for i, f in enumerate(selected_real):
    shutil.copy(f'{real_path}/{f}', f'data/images_balanced/real/{f}')
    if (i + 1) % 1000 == 0:
        print(f"  Copied {i + 1:,} real images...")

# Copy fake images
print(f"\nCopying {target_count:,} fake images...")
selected_fake = random.sample(fake_files, target_count) if fake_count > target_count else fake_files
for i, f in enumerate(selected_fake):
    shutil.copy(f'{fake_path}/{f}', f'data/images_balanced/fake/{f}')
    if (i + 1) % 1000 == 0:
        print(f"  Copied {i + 1:,} fake images...")

# Verify
balanced_real = len(os.listdir('data/images_balanced/real'))
balanced_fake = len(os.listdir('data/images_balanced/fake'))

print(f"\n{'='*60}")
print(f"✅ BALANCED DATASET CREATED!")
print(f"{'='*60}")
print(f"Location: data/images_balanced/")
print(f"  Real: {balanced_real:,}")
print(f"  Fake: {balanced_fake:,}")
print(f"  Total: {balanced_real + balanced_fake:,}")
print(f"  Balance ratio: {balanced_fake/balanced_real:.2f}:1")
print(f"{'='*60}")

print(f"\n📝 NEXT STEP:")
print(f"1. Edit training/train_image_model.py")
print(f"2. Change line 11 from:")
print(f"     data_dir = 'data/images'")
print(f"   to:")
print(f"     data_dir = 'data/images_balanced'")
print(f"\n3. Then retrain: python training/train_image_model.py")