import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys
import os

print("Loading model...")

# FIXED: Use ResNet18 to match training
model = models.resnet18(weights=None)  # Changed from resnet50
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('models/image_model.pth', map_location='cpu', weights_only=False))
model.eval()
print("✓ Model loaded\n")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get both probabilities
        fake_prob = probabilities[0][0].item()
        real_prob = probabilities[0][1].item()
        
        # Predict based on which is higher
        is_fake = fake_prob > real_prob
        confidence = max(fake_prob, real_prob)
    
    return {
        "input_type": "image",
        "is_fake": is_fake,
        "confidence": round(confidence, 4),
        "prediction": "FAKE" if is_fake else "REAL",
        "probabilities": {
            "fake": round(fake_prob, 4),
            "real": round(real_prob, 4)
        }
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python training/test_image_model.py <image_path>")
        exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        exit(1)
    
    result = predict_image(image_path)
    
    print("="*60)
    print(f"Image: {os.path.basename(image_path)}")
    print("="*60)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print("-"*60)
    print("Detailed Probabilities:")
    print(f"  FAKE: {result['probabilities']['fake']*100:.2f}%")
    print(f"  REAL: {result['probabilities']['real']*100:.2f}%")
    print("="*60)