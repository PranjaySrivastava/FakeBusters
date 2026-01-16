import os

# Check what's in the folders
real_path = "data/images/real"
fake_path = "data/images/fake"

print("Checking dataset...\n")

if os.path.exists(real_path):
    real_files = [f for f in os.listdir(real_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    print(f"✓ Real images folder exists")
    print(f"  - Contains {len(real_files)} images")
    if len(real_files) > 0:
        print(f"  - Sample files: {real_files[:3]}")
else:
    print("✗ Real images folder NOT found")

print()

if os.path.exists(fake_path):
    fake_files = [f for f in os.listdir(fake_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    print(f"✓ Fake images folder exists")
    print(f"  - Contains {len(fake_files)} images")
    if len(fake_files) > 0:
        print(f"  - Sample files: {fake_files[:3]}")
else:
    print("✗ Fake images folder NOT found")

print()
print("="*50)

total = len(real_files) + len(fake_files) if os.path.exists(real_path) and os.path.exists(fake_path) else 0

if total > 200:
    print(f"✅ Dataset ready! Total: {total} images")
    print("You can start training now!")
elif total > 0:
    print(f"⚠️ Dataset small: {total} images")
    print("You can test your code, but need more data for good results")
else:
    print("❌ Dataset is EMPTY!")
    print("Person A needs to add images to:")
    print("  - data/images/real/")
    print("  - data/images/fake/")
    