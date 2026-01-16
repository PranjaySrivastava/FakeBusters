import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys
import os

print("Loading dual detection system...")

# Model 1: Face-swap detector
faceswap_model = models.resnet18(pretrained=False)
faceswap_model.fc = nn.Linear(faceswap_model.fc.in_features, 2)
faceswap_model.load_state_dict(torch.load('models/image_model.pth', map_location='cpu', weights_only=False))
faceswap_model.eval()
print("✓ Face-swap detector loaded (87.6%)")

# Model 2: AI-gen detector  
aigen_model = models.resnet18(pretrained=False)
aigen_model.fc = nn.Linear(aigen_model.fc.in_features, 2)
aigen_model.load_state_dict(torch.load('models/aigen_model.pth', map_location='cpu', weights_only=False))
aigen_model.eval()
print("✓ AI-generation detector loaded\n")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def detect_combined(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    # Detector 1: Face-swap
    with torch.no_grad():
        outputs1 = faceswap_model(image_tensor)
        probs1 = torch.nn.functional.softmax(outputs1, dim=1)
        faceswap_fake_prob = probs1[0][0].item()
        faceswap_real_prob = probs1[0][1].item()
        faceswap_detected = faceswap_fake_prob > 0.5  # FIXED: Consistent threshold
    
    # Detector 2: AI-gen
    with torch.no_grad():
        outputs2 = aigen_model(image_tensor)
        probs2 = torch.nn.functional.softmax(outputs2, dim=1)
        aigen_fake_prob = probs2[0][0].item()
        aigen_real_prob = probs2[0][1].item()
        aigen_detected = aigen_fake_prob > 0.5
    
    # Combined decision: Use weighted approach
    # If either detector is very confident (>70%), trust it
    # Otherwise, both must agree
    
    highly_confident_fake = (faceswap_fake_prob > 0.7) or (aigen_fake_prob > 0.7)
    both_agree_fake = faceswap_detected and aigen_detected
    
    is_fake = highly_confident_fake or both_agree_fake
    
    # Confidence is the max of both detectors
    combined_confidence = max(faceswap_fake_prob, aigen_fake_prob) if is_fake else max(faceswap_real_prob, aigen_real_prob)
    
    # Determine type
    if faceswap_detected and aigen_detected:
        detection_type = "Both detectors agree: FAKE"
    elif faceswap_fake_prob > 0.7:
        detection_type = "Face-swap manipulation detected (high confidence)"
    elif aigen_fake_prob > 0.7:
        detection_type = "AI-generated image detected (high confidence)"
    elif faceswap_detected:
        detection_type = "Face-swap manipulation suspected"
    elif aigen_detected:
        detection_type = "AI-generation suspected"
    else:
        detection_type = "Both detectors agree: REAL"
    
    return {
        "input_type": "image",
        "is_fake": is_fake,
        "confidence": round(combined_confidence, 4),
        "prediction": "FAKE" if is_fake else "REAL",
        "detection_type": detection_type,
        "detectors": {
            "face_swap": {
                "fake_prob": round(faceswap_fake_prob, 4),
                "real_prob": round(faceswap_real_prob, 4),
                "detected": faceswap_detected
            },
            "ai_generation": {
                "fake_prob": round(aigen_fake_prob, 4),
                "real_prob": round(aigen_real_prob, 4),
                "detected": aigen_detected
            }
        }
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python training/test_combined_detector.py <image_path>")
        exit(1)
    
    result = detect_combined(sys.argv[1])
    
    print("="*70)
    print("DUAL DEEPFAKE DETECTOR")
    print("="*70)
    print(f"Image: {os.path.basename(sys.argv[1])}")
    print("="*70)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print(f"Type: {result['detection_type']}")
    print("-"*70)
    print("\nDetector Breakdown:")
    print(f"1. Face-Swap Detector (87.6% accuracy):")
    print(f"   FAKE: {result['detectors']['face_swap']['fake_prob']*100:.2f}%")
    print(f"   REAL: {result['detectors']['face_swap']['real_prob']*100:.2f}%")
    print(f"   Detected as fake: {result['detectors']['face_swap']['detected']}")
    print(f"\n2. AI-Generation Detector:")
    print(f"   FAKE: {result['detectors']['ai_generation']['fake_prob']*100:.2f}%")
    print(f"   REAL: {result['detectors']['ai_generation']['real_prob']*100:.2f}%")
    print(f"   Detected as fake: {result['detectors']['ai_generation']['detected']}")
    print("="*70)