import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys
import os

print("Loading IMPROVED detector...")

model = models.resnet34(weights=None)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2)
)

model.load_state_dict(torch.load('models/improved_detector.pth', map_location='cpu', weights_only=False))
model.eval()
print("✓ Improved detector loaded\n")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        fake_prob = probs[0][0].item()
        real_prob = probs[0][1].item()
        
        return {
            "prediction": "FAKE" if fake_prob > real_prob else "REAL",
            "fake_prob": round(fake_prob, 4),
            "real_prob": round(real_prob, 4),
            "confidence": round(max(fake_prob, real_prob), 4)
        }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_improved_detector.py <image_path>")
        exit(1)
    
    result = predict(sys.argv[1])
    
    print("="*70)
    print(f"Image: {os.path.basename(sys.argv[1])}")
    print("="*70)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print(f"\nProbabilities:")
    print(f"  FAKE: {result['fake_prob']*100:.2f}%")
    print(f"  REAL: {result['real_prob']*100:.2f}%")
    print("="*70)
    