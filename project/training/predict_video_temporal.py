import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import os
import json

print("Loading temporal model...")
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('models/temporal_model.pth', map_location='cpu', weights_only=False))
model.eval()
print("✓ Temporal model loaded\n")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_frame(frame):
    """Predict if a single frame is fake"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    image_tensor = transform(pil_image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        fake_prob = probabilities[0][0].item()
    return fake_prob

def analyze_video(video_path, sample_rate=5):
    """Analyze video and detect fake segments with timestamps"""
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return None
    
    print(f"Analyzing: {os.path.basename(video_path)}")
    print(f"Sample rate: Every {sample_rate} frames\n")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0 or total_frames == 0:
        print("❌ Error: Could not read video properties")
        cap.release()
        return None
    
    print(f"Video info:")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f} seconds\n")
    
    frame_predictions = []
    frame_num = 0
    
    print("Processing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_num % sample_rate == 0:
            fake_prob = predict_frame(frame)
            timestamp = frame_num / fps
            
            frame_predictions.append({
                'frame': frame_num,
                'timestamp': timestamp,
                'fake_probability': fake_prob,
                'is_fake': fake_prob > 0.5
            })
            
            if len(frame_predictions) % 20 == 0:
                print(f"  Processed {len(frame_predictions)} frames...", end='\r')
        
        frame_num += 1
    
    cap.release()
    print(f"\n✓ Processed {len(frame_predictions)} frames\n")
    
    # Calculate overall statistics
    fake_count = sum(1 for p in frame_predictions if p['is_fake'])
    total_analyzed = len(frame_predictions)
    overall_confidence = fake_count / total_analyzed if total_analyzed > 0 else 0
    
    # Find manipulated segments
    manipulated_segments = find_segments(frame_predictions, fps)
    
    result = {
        "input_type": "video",
        "video_is_fake": overall_confidence > 0.5,
        "overall_confidence": round(overall_confidence, 4),
        "frames_analyzed": total_analyzed,
        "fake_frames": fake_count,
        "manipulated_segments": manipulated_segments
    }
    
    return result

def find_segments(predictions, fps, min_segment_length=1.0):
    """Group consecutive fake frames into segments"""
    segments = []
    current_segment = None
    
    for pred in predictions:
        if pred['is_fake']:
            if current_segment is None:
                # Start new segment
                current_segment = {
                    'start_frame': pred['frame'],
                    'start_time': pred['timestamp'],
                    'end_frame': pred['frame'],
                    'end_time': pred['timestamp'],
                    'confidence_sum': pred['fake_probability'],
                    'count': 1
                }
            else:
                # Extend current segment
                current_segment['end_frame'] = pred['frame']
                current_segment['end_time'] = pred['timestamp']
                current_segment['confidence_sum'] += pred['fake_probability']
                current_segment['count'] += 1
        else:
            # Real frame - close current segment if exists
            if current_segment is not None:
                duration = current_segment['end_time'] - current_segment['start_time']
                
                # Only add if segment is long enough
                if duration >= min_segment_length:
                    avg_confidence = current_segment['confidence_sum'] / current_segment['count']
                    
                    segments.append({
                        'start_time': format_timestamp(current_segment['start_time']),
                        'end_time': format_timestamp(current_segment['end_time']),
                        'confidence': round(avg_confidence, 4)
                    })
                
                current_segment = None
    
    # Close last segment if exists
    if current_segment is not None:
        duration = current_segment['end_time'] - current_segment['start_time']
        if duration >= min_segment_length:
            avg_confidence = current_segment['confidence_sum'] / current_segment['count']
            segments.append({
                'start_time': format_timestamp(current_segment['start_time']),
                'end_time': format_timestamp(current_segment['end_time']),
                'confidence': round(avg_confidence, 4)
            })
    
    return segments

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# ========================================
# MAIN
# ========================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("="*70)
        print("DEEPFAKE VIDEO ANALYZER - Temporal Model")
        print("="*70)
        print("\nUsage: python training/predict_video_temporal.py <video_path>")
        print("\nExample:")
        print('  python training/predict_video_temporal.py "C:\\path\\to\\video.mp4"')
        print("="*70)
        exit(1)
    
    video_path = sys.argv[1]
    
    print("="*70)
    print("DEEPFAKE VIDEO ANALYZER - Temporal Model (85.8% Accuracy)")
    print("="*70)
    print()
    
    result = analyze_video(video_path, sample_rate=5)
    
    if result:
        print("="*70)
        print("RESULTS")
        print("="*70)
        print(f"Video is FAKE: {result['video_is_fake']}")
        print(f"Overall Confidence: {result['overall_confidence']*100:.2f}%")
        print(f"Frames Analyzed: {result['frames_analyzed']}")
        print(f"Fake Frames: {result['fake_frames']}")
        print()
        
        if result['manipulated_segments']:
            print("Manipulated Segments:")
            print("-" * 70)
            for i, seg in enumerate(result['manipulated_segments'], 1):
                print(f"{i}. {seg['start_time']} → {seg['end_time']} "
                      f"(Confidence: {seg['confidence']*100:.2f}%)")
        else:
            print("No manipulated segments detected")
            print("(All frames appear authentic)")
        
        print("="*70)
        print("\nJSON Output:")
        print(json.dumps(result, indent=2))
        print("="*70)