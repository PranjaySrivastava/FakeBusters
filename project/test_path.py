import os

# Test different paths
paths_to_test = [
    r"C:\Users\Pranjay\Desktop\auraverse 2.0\required",
    r"C:\Users\Pranjay\Desktop\auraverse 2.0\required\required",
]

print("Testing paths...\n")

for path in paths_to_test:
    print(f"Testing: {path}")
    if os.path.exists(path):
        print("  ✅ Path exists!")
        contents = os.listdir(path)
        print(f"  Contains: {contents}")
        
        # Check for our folders
        has_original = "original" in contents
        has_deepfakes = "Deepfakes" in contents
        has_faceswap = "FaceSwap" in contents
        
        if has_original and has_deepfakes and has_faceswap:
            print("  🎯 THIS IS THE CORRECT PATH!")
            print(f"\n  Use this in organize_quick_start.py line 12:")
            print(f'  source_base = r"{path}"')
        else:
            print(f"  ⚠️  Missing folders")
    else:
        print("  ❌ Path does NOT exist")
    print()