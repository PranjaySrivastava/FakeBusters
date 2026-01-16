# Deepfake Detection System

## Overview
Complete deepfake detection system with image classification and video timestamp localization.

## Team
- **Person A**: Data preparation and preprocessing
- **Person B**: Model training and development
- **Person C**: Integration and demo interface

## Features
- ✅ Image deepfake detection (87.6% accuracy)
- ✅ Video temporal analysis with timestamp localization
- ✅ Confidence scores for all predictions
- ✅ JSON output format for easy integration

## Models
### 1. Face-Swap Detector
- **Architecture**: ResNet18
- **Accuracy**: 87.6%
- **Dataset**: FaceForensics++ (20,000 balanced images)
- **Training**: 15 epochs

### 2. Video Temporal Analyzer
- **Method**: Frame-by-frame analysis
- **Output**: Manipulated segments with timestamps

## Project Structure
```
project/
├── data/                    # Datasets (not in repo - too large)
├── models/                  # Trained models (not in repo - too large)
├── training/
│   ├── train_image_model.py       # Train image detector
│   ├── test_image_model.py        # Test image detector
│   ├── predict_video.py           # Video analysis with timestamps
│   └── train_temporal_model.py    # Temporal model training
├── balance_dataset.py       # Dataset balancing script
├── check_dataset.py         # Verify dataset
├── extract_frames_quick.py  # Extract frames from videos
├── organize_quick_start.py  # Organize FaceForensics dataset
├── requirements.txt         # Dependencies
└── README.md
```

## Installation

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd project
```

### 2. Create Virtual Environment
```bash
conda create -n deepfake_env python=3.9
conda activate deepfake_env
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
Download FaceForensics++ from: https://github.com/ondyari/FaceForensics

### 5. Organize Dataset
```bash
python extract_frames_quick.py
python balance_dataset.py
```

### 6. Train Models
```bash
python training/train_image_model.py
```

## Usage

### Image Detection
```bash
python training/test_image_model.py <image_path>
```

**Output:**
```json
{
  "input_type": "image",
  "is_fake": true,
  "confidence": 0.89,
  "prediction": "FAKE"
}
```

### Video Detection with Timestamps
```bash
python training/predict_video.py <video_path>
```

**Output:**
```json
{
  "input_type": "video",
  "video_is_fake": true,
  "overall_confidence": 0.9875,
  "manipulated_segments": [
    {
      "start_time": "00:00:00",
      "end_time": "00:00:08",
      "confidence": 0.9465
    }
  ]
}
```

## Performance

### Metrics
- Validation Accuracy: 87.6%
- Training Accuracy: 90.6%
- Dataset: 20,000 images (balanced)

### Example Results
- Face-swap deepfakes: 85-90% detection accuracy
- Video analysis: Accurate timestamp localization

## Requirements
See `requirements.txt`

## Model Files
**Note**: Model files are too large for GitHub. Download from:
- [Google Drive Link] (add your link after uploading)
- Or train from scratch using instructions above

## Limitations
- Optimized for face-swap deepfakes (FaceForensics++ dataset)
- May struggle with modern AI-generated images
- Best performance on similar data to training set

## Future Improvements
- [ ] Add AI-generated image detection
- [ ] Implement ensemble methods
- [ ] Real-time video processing
- [ ] Web interface

## License
Educational project - 2026

## Acknowledgments
- FaceForensics++ dataset
- PyTorch team
- ResNet architecture