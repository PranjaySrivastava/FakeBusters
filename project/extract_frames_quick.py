import cv2
import os
from pathlib import Path

# ========================================
# CONFIGURATION
# ========================================

# Your path (same as before)
source_base = r"C:\Users\Pranjay\Desktop\auraverse 2.0\required\required"

# Where to save extracted frames
target_base = "data/images"

# Folders to process
REAL_FOLDER = "original"
FAKE_FOLDERS = ["Deepfakes", "FaceSwap"]

# How many frames to extract per video?
FRAMES_PER_VIDEO = 10  # Adjust: 10 = fast, 30 = more data but slower

# ========================================
# Setup
# ========================================

os.makedirs(f"{target_base}/real", exist_ok=True)
os.makedirs(f"{target_base}/fake", exist_ok=True)

frame_count = {'real': 0, 'fake': 0}
video_count = {'real': 0, 'fake': 0}

print("="*70)
print("VIDEO FRAME EXTRACTOR")
print(f"Extracting {FRAMES_PER_VIDEO} frames per video")
print("="*70)
print(f"Source: {source_base}")
print(f"Target: {target_base}")
print("="*70)

# ========================================
# Frame extraction function
# ========================================

def extract_frames_from_video(video_path, output_folder, label):
    """Extract frames from a single video"""
    global frame_count
    
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return 0
        
        # Calculate which frames to extract (evenly spaced)
        frame_indices = [int(total_frames * i / FRAMES_PER_VIDEO) 
                         for i in range(FRAMES_PER_VIDEO)]
        
        video_name = Path(video_path).stem  # Get filename without extension
        extracted = 0
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Save frame as image
                output_filename = f"{label}_{frame_count[label]:06d}_{video_name}_frame{frame_idx}.jpg"
                output_path = os.path.join(output_folder, output_filename)
                
                cv2.imwrite(output_path, frame)
                frame_count[label] += 1
                extracted += 1
        
        cap.release()
        return extracted
        
    except Exception as e:
        print(f"      Error processing video: {e}")
        return 0

# ========================================
# Process folder function
# ========================================

def process_video_folder(folder_path, target_folder, label):
    """Process all videos in a folder"""
    
    if not os.path.exists(folder_path):
        print(f"\n❌ Folder not found: {folder_path}")
        return
    
    print(f"\n📁 Processing: {os.path.basename(folder_path)}")
    print(f"   Looking for videos...")
    
    # Find all video files
    video_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append(os.path.join(root, file))
    
    if len(video_files) == 0:
        print(f"   ⚠️  No video files found!")
        return
    
    print(f"   Found {len(video_files)} videos")
    print(f"   Extracting frames (this may take a few minutes)...")
    
    start_count = frame_count[label]
    
    # Process each video
    for i, video_path in enumerate(video_files, 1):
        extracted = extract_frames_from_video(video_path, target_folder, label)
        video_count[label] += 1
        
        # Progress update every 10 videos
        if i % 10 == 0 or i == len(video_files):
            print(f"      ✓ Processed {i}/{len(video_files)} videos, "
                  f"{frame_count[label] - start_count} frames extracted")
    
    print(f"   ✅ Completed: {frame_count[label] - start_count} frames from {video_count[label]} videos")

# ========================================
# MAIN PROCESSING
# ========================================

print("\n" + "="*70)
print("STEP 1/3: Extracting from 'original' folder (REAL)")
print("="*70)

real_folder = os.path.join(source_base, REAL_FOLDER)
process_video_folder(real_folder, f"{target_base}/real", 'real')

print("\n" + "="*70)
print("STEP 2/3: Extracting from 'Deepfakes' folder (FAKE)")
print("="*70)

deepfakes_folder = os.path.join(source_base, FAKE_FOLDERS[0])
process_video_folder(deepfakes_folder, f"{target_base}/fake", 'fake')

print("\n" + "="*70)
print("STEP 3/3: Extracting from 'FaceSwap' folder (FAKE)")
print("="*70)

faceswap_folder = os.path.join(source_base, FAKE_FOLDERS[1])
process_video_folder(faceswap_folder, f"{target_base}/fake", 'fake')

# ========================================
# FINAL SUMMARY
# ========================================

print("\n" + "="*70)
print("✅ FRAME EXTRACTION COMPLETE!")
print("="*70)
print(f"📊 Real frames:  {frame_count['real']:,} (from {video_count['real']} videos)")
print(f"📊 Fake frames:  {frame_count['fake']:,} (from {video_count['fake']} videos)")
print(f"📊 Total frames: {frame_count['real'] + frame_count['fake']:,}")
print(f"\n📂 Data saved to:")
print(f"   {os.path.abspath(target_base)}/real/")
print(f"   {os.path.abspath(target_base)}/fake/")
print("="*70)

if frame_count['real'] > 100 and frame_count['fake'] > 100:
    print("\n🎉 Dataset ready for training!")
    print("\n🚀 Next steps:")
    print("   1. Verify: python check_dataset.py")
    print("   2. Train:  python training/train_image_model.py")
elif frame_count['real'] > 0 or frame_count['fake'] > 0:
    print("\n⚠️  Dataset is small but you can start testing")
    print("   🚀 Try: python training/train_image_model.py")
else:
    print("\n❌ No frames extracted!")
    print("   Check if folders contain video files")
print("="*70)