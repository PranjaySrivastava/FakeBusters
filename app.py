from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import os
import base64
import io
import tempfile

app = Flask(__name__)
CORS(app)

print("Loading models...")

# Image Model (Deepfake Detection)
image_model = models.resnet18(pretrained=False)
image_model.fc = nn.Linear(image_model.fc.in_features, 2)
if os.path.exists('models/image_model.pth'):
    image_model.load_state_dict(torch.load('models/image_model.pth', map_location='cpu', weights_only=False))
    image_model.eval()
    print("✓ Image model loaded (Deepfake Detection)")
else:
    image_model = None
    print("⚠ Image model not found")

# Improved Detector Model (AI-Generated Detection)
aigen_model = models.resnet34(weights=None)
num_features = aigen_model.fc.in_features
aigen_model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2)
)

if os.path.exists('models/improved_detector.pth'):
    aigen_model.load_state_dict(torch.load('models/improved_detector.pth', map_location='cpu', weights_only=False))
    aigen_model.eval()
    print("✓ Improved Detector loaded (AI-Generated Detection)")
else:
    aigen_model = None
    print("⚠ Improved Detector not found")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_frame(model, frame):
    """Predict if a single frame is fake"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    image_tensor = transform(pil_image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        fake_prob = probabilities[0][0].item()
    return fake_prob

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def find_segments(predictions, fps, min_segment_length=1.0):
    """Group consecutive fake frames into segments"""
    segments = []
    current_segment = None
    
    for pred in predictions:
        if pred['is_fake']:
            if current_segment is None:
                current_segment = {
                    'start_time': pred['timestamp'],
                    'end_time': pred['timestamp'],
                    'confidence_sum': pred['fake_probability'],
                    'count': 1
                }
            else:
                current_segment['end_time'] = pred['timestamp']
                current_segment['confidence_sum'] += pred['fake_probability']
                current_segment['count'] += 1
        else:
            if current_segment is not None:
                duration = current_segment['end_time'] - current_segment['start_time']
                if duration >= min_segment_length:
                    avg_confidence = current_segment['confidence_sum'] / current_segment['count']
                    segments.append({
                        'start_time': format_timestamp(current_segment['start_time']),
                        'end_time': format_timestamp(current_segment['end_time']),
                        'confidence': round(avg_confidence * 100, 2)
                    })
                current_segment = None
    
    if current_segment is not None:
        duration = current_segment['end_time'] - current_segment['start_time']
        if duration >= min_segment_length:
            avg_confidence = current_segment['confidence_sum'] / current_segment['count']
            segments.append({
                'start_time': format_timestamp(current_segment['start_time']),
                'end_time': format_timestamp(current_segment['end_time']),
                'confidence': round(avg_confidence * 100, 2)
            })
    
    return segments

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if server is running and which models are available"""
    return jsonify({
        'status': 'ok',
        'models': {
            'image_model': image_model is not None,
            'aigen_model': aigen_model is not None
        }
    })

@app.route('/api/analyze/image', methods=['POST'])
def analyze_image():
    """Analyze image for deepfake detection"""
    if not image_model:
        return jsonify({'error': 'Image model not loaded'}), 500
    
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = image_model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            fake_prob = probabilities[0][0].item()
            real_prob = probabilities[0][1].item()
        
        is_fake = fake_prob > real_prob
        confidence = max(fake_prob, real_prob)
        
        return jsonify({
            'prediction': 'Deepfake Detected' if is_fake else 'Authentic Media Detected',
            'confidence': round(confidence * 100, 2),
            'is_fake': is_fake,
            'probabilities': {
                'fake': round(fake_prob * 100, 2),
                'real': round(real_prob * 100, 2)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/video', methods=['POST'])
def analyze_video():
    """Analyze video for deepfake detection"""
    if not image_model:
        return jsonify({'error': 'Video model not loaded'}), 500
    
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            video_file.save(tmp.name)
            tmp_path = tmp.name
        
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps == 0 or total_frames == 0:
            os.unlink(tmp_path)
            return jsonify({'error': 'Could not read video'}), 400
        
        frame_predictions = []
        frame_num = 0
        sample_rate = 5
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_num % sample_rate == 0:
                fake_prob = predict_frame(image_model, frame)
                timestamp = frame_num / fps
                
                frame_predictions.append({
                    'frame': frame_num,
                    'timestamp': timestamp,
                    'fake_probability': fake_prob,
                    'is_fake': fake_prob > 0.5
                })
            
            frame_num += 1
        
        cap.release()
        os.unlink(tmp_path)
        
        fake_count = sum(1 for p in frame_predictions if p['is_fake'])
        total_analyzed = len(frame_predictions)
        overall_confidence = fake_count / total_analyzed if total_analyzed > 0 else 0
        
        segments = find_segments(frame_predictions, fps)
        
        timeline_segments = []
        for pred in frame_predictions:
            if pred['is_fake']:
                position_percent = (pred['frame'] / total_frames) * 100
                timeline_segments.append({
                    'position': round(position_percent, 2),
                    'confidence': round(pred['fake_probability'] * 100, 2)
                })
        
        return jsonify({
            'prediction': 'Deepfake Detected' if overall_confidence > 0.5 else 'Authentic Media Detected',
            'confidence': round(overall_confidence * 100, 2),
            'is_fake': overall_confidence > 0.5,
            'frames_analyzed': total_analyzed,
            'fake_frames': fake_count,
            'segments': segments,
            'timeline': timeline_segments
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/ai', methods=['POST'])
def analyze_ai():
    """Analyze image for AI-generation detection using Improved Detector"""
    if not aigen_model:
        return jsonify({'error': 'AI-Gen model (Improved Detector) not loaded'}), 500
    
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = aigen_model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            ai_prob = probabilities[0][0].item()
            real_prob = probabilities[0][1].item()
        
        is_ai = ai_prob > real_prob
        confidence = max(ai_prob, real_prob)
        
        return jsonify({
            'prediction': 'AI-Generated Detected' if is_ai else 'Real Image Detected',
            'confidence': round(confidence * 100, 2),
            'is_ai': is_ai,
            'probabilities': {
                'ai': round(ai_prob * 100, 2),
                'real': round(real_prob * 100, 2)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("FakeBusters Backend Server")
    print("="*70)
    print("Server starting on http://localhost:5000")
    print(f"✓ Image Model (Deepfake): {'Loaded' if image_model else 'Not Found'}")
    print(f"✓ AI-Gen Model (Improved Detector): {'Loaded' if aigen_model else 'Not Found'}")
    print("="*70 + "\n")
    app.run(debug=True, port=5000)