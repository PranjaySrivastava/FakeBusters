# 🔍 AuraVerse 2.0 - Deepfake Detection System

Advanced AI-powered deepfake and AI-generated content detection using deep learning.

## 🌟 Features

- **Universal Detector** - Detects ANY photo type (phone, screenshot, professional) - 82%+ accuracy
- **AI-Generated Detection** - Identifies AI-created images - 97% accuracy
- **Face-Swap Detection** - Detects deepfake face swaps - 87.6% accuracy
- **Video Analysis** - Frame-by-frame detection with timestamps
- **Ensemble Detection** - Combines multiple models for maximum accuracy

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/PranjaySrivastava/FakeBusters.git

cd auraverse-2.0/project

# Create virtual environment
python -m venv deepfake_env
deepfake_env\Scripts\activate  # Windows
# source deepfake_env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Download Datasets

1. **CIFAKE Dataset** - [Download from Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
2. Extract to `data/cifake_raw/`
3. Organize: `python organize_cifake_simple.py`

### Download Pre-trained Models

**Models are too large for GitHub. Download separately:**
- Models are not included in this repository due to size limitations

## 🎯 Usage

### Detect Images
```bash
# Best for ANY random photo (recommended)
python test_improved_detector.py "your_image.jpg"

# AI-generated detection
python test_aigen_model_enhanced.py "your_image.jpg"

# Ensemble (all models combined)
python test_combined_detector.py "your_image.jpg"
```

### Analyze Videos
```bash
python training/predict_video.py "your_video.mp4"
```

## 🎓 Training Models

### Train Universal Detector (Recommended)
```bash
python train_improved_simple.py
```
**Time:** ~20 minutes | **Output:** `models/improved_detector.pth`

### Train AI-Gen Detector
```bash
python train_aigen_model.py
```
**Accuracy:** 97% | **Time:** ~15 minutes

### Train Face-Swap Detector
```bash
python train_image_model.py
```
**Accuracy:** 87.6% | **Time:** ~20 minutes

## 📊 Results

| Model | Purpose | Accuracy |
|-------|---------|----------|
| Universal Detector | Any photo type | 82-85% |
| AI-Gen Detector | AI-generated images | 97.1% |
| Face-Swap Detector | Deepfake faces | 87.6% |

## 📁 Project Structure
```
project/
├── models/              # Trained models (download separately)
├── data/                # Datasets (download separately)
├── training/            # Training & prediction scripts
├── test_improved_detector.py      # Main detector
├── train_improved_simple.py       # Main training script
└── requirements.txt
```

## 🛠️ Troubleshooting

**Model says all images are fake?**
- Use `test_improved_detector.py` instead
- This model handles phone photos better

**GPU out of memory?**
- Reduce `batch_size` in training scripts to 16 or 8

## 📧 Contact

- GitHub: [@PranjaySrivastava](https://github.com/PranjaySrivastava)
- Email: pranjay950@gmail.com
```

Make sure it has:
```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
tqdm>=4.65.0
matplotlib>=3.7.0
```