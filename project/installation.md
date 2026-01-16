# 🚀 FakeBusters Setup Guide

## Complete guide for cloning and running the FakeBusters deepfake detection system

---

## 📋 Prerequisites

Before starting, ensure you have:
- Python 3.9 or 3.10 installed
- Git installed
- 10+ GB free disk space (for dataset and models)
- (Optional) NVIDIA GPU with CUDA for faster training

---

## 🔧 Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/PranjaySrivastava/FakeBusters.git

# Navigate into the project
cd FakeBusters
```

---

## 🐍 Step 2: Set Up Python Environment

### Option A: Using Conda (Recommended)

```bash
# Create a new environment
conda create -n fakebusters python=3.9

# Activate the environment
conda activate fakebusters

# Install dependencies
pip install -r requirements.txt
```

### Option B: Using venv

```bash
# Create virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Step 3: Get the Dataset

### Option 1: Download FaceForensics++ Dataset (Recommended)

1. Visit: https://github.com/ondyari/FaceForensics
2. Request access to the dataset
3. Download the following folders:
   - `original` (real videos/images)
   - `Deepfakes` (fake videos/images)
   - `FaceSwap` (fake videos/images)

4. Place them in your project directory

### Option 2: Use Your Own Dataset

Create this folder structure:
```
FakeBusters/
├── data/
│   └── images/
│       ├── real/     # Put real images here
│       └── fake/     # Put fake images here
```

---

## 🎬 Step 4: Prepare the Data

### If you have videos (FaceForensics++):

1. **Update the path** in `extract_frames_quick.py` (line 12):
   ```python
   source_base = r"C:\path\to\your\downloaded\faceforensics"
   ```

2. **Extract frames** from videos:
   ```bash
   python extract_frames_quick.py
   ```
   This creates: `data/images/real/` and `data/images/fake/`

3. **Balance the dataset** (optional but recommended):
   ```bash
   python balance_dataset.py
   ```
   This creates: `data/images_balanced/`

### If you have images already:

1. Place images directly in:
   - `data/images/real/` - Real images
   - `data/images/fake/` - Fake images

2. **Verify** your dataset:
   ```bash
   python check_dataset.py
   ```

---

## 🏋️ Step 5: Train the Models

### Train Image Model

```bash
python training/train_image_model.py
```

**What this does:**
- Trains a ResNet18 model on images
- Takes ~15-30 minutes (depending on GPU)
- Saves model to `models/image_model.pth`
- Expected accuracy: ~87-90%

### Train Temporal Model (Optional - for video analysis)

```bash
python training/train_temporal_model.py
```

**What this does:**
- Trains an LSTM-based temporal model
- Analyzes frame sequences
- Takes ~30-60 minutes
- Saves model to `models/temporal_model.pth`

---

## 🧪 Step 6: Test the Models

### Test on a Single Image

```bash
python training/test_image_model.py "path/to/test/image.jpg"
```

**Example output:**
```
============================================================
Image: test_image.jpg
============================================================
Prediction: FAKE
Confidence: 89.23%
------------------------------------------------------------
Detailed Probabilities:
  FAKE: 89.23%
  REAL: 10.77%
============================================================
```

### Test on a Video

```bash
python training/predict_video.py "path/to/test/video.mp4"
```

**Example output:**
```
============================================================
RESULTS
============================================================
Video is FAKE: True
Overall Confidence: 92.50%
Frames Analyzed: 120
Fake Frames: 111

Manipulated Segments:
----------------------------------------------------------------------
1. 00:00:02 → 00:00:08 (Confidence: 94.65%)
============================================================
```

---

## 📦 Important: Model Files

⚠️ **Model files are NOT included in the repository** (they're too large for GitHub)

You have two options:

### Option 1: Train from Scratch (Recommended)
Follow Step 5 above to train your own models

### Option 2: Download Pre-trained Models
If the repository owner has shared pre-trained models:
1. Download `image_model.pth` and `temporal_model.pth`
2. Create a `models/` folder in your project
3. Place the `.pth` files inside:
   ```
   FakeBusters/
   └── models/
       ├── image_model.pth
       └── temporal_model.pth
   ```

---

## 🗂️ Project Structure

After setup, your project should look like this:

```
FakeBusters/
├── data/
│   ├── images/              # Original dataset
│   │   ├── real/           # Real images
│   │   └── fake/           # Fake images
│   └── images_balanced/    # Balanced dataset (optional)
│       ├── real/
│       └── fake/
├── models/
│   ├── image_model.pth     # Trained image model
│   └── temporal_model.pth  # Trained temporal model (optional)
├── training/
│   ├── train_image_model.py
│   ├── test_image_model.py
│   ├── predict_video.py
│   └── train_temporal_model.py
├── balance_dataset.py
├── check_dataset.py
├── extract_frames_quick.py
├── requirements.txt
└── README.md
```

---

## 🐛 Troubleshooting

### Issue: "No module named 'torch'"
**Solution:** Make sure your virtual environment is activated and dependencies are installed:
```bash
conda activate fakebusters  # or: source venv/bin/activate
pip install -r requirements.txt
```

### Issue: "Data directory not found"
**Solution:** Make sure you've run the data preparation scripts and the folders exist:
```bash
python check_dataset.py
```

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size in training scripts:
- Open `training/train_image_model.py`
- Change line 13: `batch_size = 16` (reduce from 32)

### Issue: "OpenCV error when processing videos"
**Solution:** Install codec support:
```bash
# Windows
pip install opencv-python-headless==4.11.0.86

# Linux
sudo apt-get install ffmpeg
```

### Issue: Models not loading
**Solution:** Make sure model files exist:
```bash
# Check if models folder exists
ls models/

# If not, create it and train models
mkdir models
python training/train_image_model.py
```

---

## Quick Commands Reference

```bash
# Activate environment
conda activate fakebusters

# Check dataset
python check_dataset.py

# Extract frames from videos
python extract_frames_quick.py

# Balance dataset
python balance_dataset.py

# Train image model
python training/train_image_model.py

# Test on image
python training/test_image_model.py "image.jpg"

# Test on video
python training/predict_video.py "video.mp4"

# Train temporal model (optional)
python training/train_temporal_model.py

# Deactivate environment when done
conda deactivate
```

---

## Expected Results

After successful setup:
- ✅ Image model accuracy: **87-90%**
- ✅ Video detection with timestamp localization
- ✅ JSON output for easy integration
- ✅ Fast inference: ~0.1s per image, ~5s per video

---

## Getting Help

If you encounter issues:
1. Check the **Troubleshooting** section above
2. Verify your Python version: `python --version`
3. Check installed packages: `pip list`
4. Open an issue on GitHub with error details

---

## 🎉 You're All Set!

Once everything is set up, you can:
- Train models on your own dataset
- Test detection on images and videos
- Integrate into your own applications
- Experiment with different architectures

Happy detecting! 🔍
