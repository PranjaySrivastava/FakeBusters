import os
import shutil
from pathlib import Path

# ========================================
# CONFIGURATION
# ========================================

# WHERE ARE YOUR FACEFORENSICS FOLDERS?
# Find the path by right-clicking on one folder → Properties → Location
# Example: "C:\Users\YourName\Downloads\FaceForensics"
source_base = r"C:\Users\Pranjay\Desktop\auraverse 2.0\required\required"
  # ← CHANGE THIS PATH!

# Target location (where training code expects data)
target_base = "data/images"

# ========================================
# Folders we're using
# ========================================

REAL_FOLDER = "original"
FAKE_FOLDERS = ["Deepfakes", "FaceSwap"]

# ========================================
# Create target folders
# ========================================

os.makedirs(f"{target_base}/real", exist_ok=True)
os.makedirs(f"{target_base}/fake", exist_ok=True)

copied_real = 0
copied_fake = 0

print("="*60)
print("Quick Start Dataset Organizer")
print("Using: original + Deepfakes + FaceSwap")
print("="*60)

# ========================================
# Helper function
# ========================================

def copy_images_from_folder(source_folder, target_folder, label):
    """Copy all images from source to target"""
    global copied_real, copied_fake
    
    if not os.path.exists(source_folder):
        print(f"\n❌ ERROR: Folder not found!")
        print(f"   Looking for: {source_folder}")
        print(f"   Please check the path in line 10 of this script")
        return
    
    print(f"\n📁 Processing: {os.path.basename(source_folder)}")
    print(f"   Source: {source_folder}")
    
    found_files = 0
    
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # Check if it's an image file
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                source_path = os.path.join(root, file)
                
                # Create unique filename
                folder_name = os.path.basename(source_folder)
                
                if label == 'real':
                    target_filename = f"real_{copied_real:06d}_{file}"
                    target_path = os.path.join(target_folder, target_filename)
                    copied_real += 1
                else:
                    target_filename = f"fake_{copied_fake:06d}_{folder_name}_{file}"
                    target_path = os.path.join(target_folder, target_filename)
                    copied_fake += 1
                
                # Copy file
                try:
                    shutil.copy2(source_path, target_path)
                    found_files += 1
                    
                    # Progress indicator every 100 files
                    if found_files % 100 == 0:
                        print(f"   ✓ Copied {found_files} images...")
                        
                except Exception as e:
                    print(f"   ⚠️  Error copying {file}: {e}")
    
    if found_files == 0:
        print(f"   ⚠️  No images found in this folder!")
        print(f"   Check if it contains .mp4 videos instead")
    else:
        print(f"   ✅ Completed: {found_files} images copied")

# ========================================
# Process REAL images
# ========================================

print("\n" + "="*60)
print("STEP 1/3: Copying REAL images from 'original' folder")
print("="*60)

real_folder_path = os.path.join(source_base, REAL_FOLDER)
copy_images_from_folder(real_folder_path, f"{target_base}/real", 'real')

# ========================================
# Process FAKE images
# ========================================

print("\n" + "="*60)
print("STEP 2/3: Copying FAKE images from 'Deepfakes' folder")
print("="*60)

deepfakes_path = os.path.join(source_base, FAKE_FOLDERS[0])
copy_images_from_folder(deepfakes_path, f"{target_base}/fake", 'fake')

print("\n" + "="*60)
print("STEP 3/3: Copying FAKE images from 'FaceSwap' folder")
print("="*60)

faceswap_path = os.path.join(source_base, FAKE_FOLDERS[1])
copy_images_from_folder(faceswap_path, f"{target_base}/fake", 'fake')

# ========================================
# Final Summary
# ========================================

print("\n" + "="*60)
print("✅ ORGANIZATION COMPLETE!")
print("="*60)
print(f"📊 Real images: {copied_real}")
print(f"📊 Fake images: {copied_fake}")
print(f"📊 Total: {copied_real + copied_fake}")
print(f"\n📂 Organized data location:")
print(f"   {os.path.abspath(target_base)}/real/")
print(f"   {os.path.abspath(target_base)}/fake/")
print("="*60)

# ========================================
# Next steps advice
# ========================================

if copied_real == 0 or copied_fake == 0:
    print("\n⚠️  WARNING: No images were copied!")
    print("\n🔍 Troubleshooting:")
    print("   1. Check if source_base path is correct (line 10)")
    print("   2. Check if folders contain .jpg files or .mp4 videos")
    print("   3. If videos, you need frame extraction script")
    print("\nTo check:")
    print(f"   dir {source_base}\\{REAL_FOLDER}")
    
elif copied_real < 100 or copied_fake < 100:
    print("\n⚠️  Dataset is small but usable for testing")
    print("   You can train, but results might not be great")
    print("\n🚀 Next step: python training/train_image_model.py")
    
else:
    print("\n🎉 Great! Dataset is ready!")
    print("\n🚀 Next steps:")
    print("   1. Verify: python check_dataset.py")
    print("   2. Train:  python training/train_image_model.py")