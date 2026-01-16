import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys
import os
import numpy as np

print("Loading AI-generated image detector...")
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)

if not os.path.exists('models/aigen_model.pth'):
    print("❌ Model not found: models/aigen_model.pth")
    print("   Please train it first: python train_aigen_model.py")
    exit(1)

model.load_state_dict(torch.load('models/aigen_model.pth', map_location='cpu', weights_only=False))
model.eval()
print("✓ Model loaded\n")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path, verbose=True):
    """Enhanced prediction with detailed analysis"""
    image = Image.open(image_path).convert('RGB')
    
    if verbose:
        print(f"Image Info:")
        print(f"  Original size: {image.size}")
        print(f"  Mode: {image.mode}")
        print()
    
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        
        if verbose:
            print(f"Raw logits: {outputs[0].numpy()}")
        
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Class 0 = fake/AI, Class 1 = real
        ai_generated_prob = probabilities[0][0].item()
        real_prob = probabilities[0][1].item()
        
        is_ai_generated = ai_generated_prob > real_prob
        confidence = max(ai_generated_prob, real_prob)
        margin = abs(ai_generated_prob - real_prob)
    
    return {
        "input_type": "image",
        "is_ai_generated": is_ai_generated,
        "confidence": round(confidence, 4),
        "prediction": "AI-GENERATED" if is_ai_generated else "REAL",
        "probabilities": {
            "ai_generated": round(ai_generated_prob, 4),
            "real": round(real_prob, 4)
        },
        "margin": round(margin, 4),
        "certainty": "HIGH" if margin > 0.3 else "MEDIUM" if margin > 0.1 else "LOW"
    }

def batch_test(folder_path):
    """Test all images in a folder"""
    if not os.path.isdir(folder_path):
        print(f"❌ Not a directory: {folder_path}")
        return
    
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"❌ No images found in {folder_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"BATCH TESTING: {len(image_files)} images")
    print(f"{'='*70}\n")
    
    results = []
    for img_file in image_files[:50]:
        img_path = os.path.join(folder_path, img_file)
        try:
            result = predict_image(img_path, verbose=False)
            results.append((img_file, result))
            
            pred = result['prediction']
            conf = result['confidence'] * 100
            cert = result['certainty']
            print(f"{img_file[:40]:40s} | {pred:15s} | {conf:5.1f}% | {cert}")
        except Exception as e:
            print(f"{img_file[:40]:40s} | ERROR: {e}")
    
    if results:
        ai_count = sum(1 for _, r in results if r['is_ai_generated'])
        real_count = len(results) - ai_count
        avg_confidence = np.mean([r['confidence'] for _, r in results])
        
        print(f"\n{'='*70}")
        print("SUMMARY:")
        print(f"  AI-Generated: {ai_count}/{len(results)}")
        print(f"  Real: {real_count}/{len(results)}")
        print(f"  Avg Confidence: {avg_confidence*100:.1f}%")
        print(f"{'='*70}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("="*70)
        print("AI-GENERATED IMAGE DETECTOR - Enhanced Version")
        print("="*70)
        print("\nUsage:")
        print("  Single image: python test_aigen_model_enhanced.py <image_path>")
        print("  Batch test:   python test_aigen_model_enhanced.py <folder_path>")
        print("\nExamples:")
        print('  python test_aigen_model_enhanced.py "test.jpg"')
        print('  python test_aigen_model_enhanced.py "data/test_images/"')
        print("="*70)
        exit(1)
    
    path = sys.argv[1]
    
    if os.path.isdir(path):
        batch_test(path)
    elif os.path.isfile(path):
        result = predict_image(path, verbose=True)
        
        print("="*70)
        print("AI-GENERATED IMAGE DETECTOR")
        print("="*70)
        print(f"Image: {os.path.basename(path)}")
        print("="*70)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print(f"Certainty: {result['certainty']} (margin: {result['margin']:.3f})")
        print("-"*70)
        print("Detailed Probabilities:")
        print(f"  AI-GENERATED: {result['probabilities']['ai_generated']*100:.2f}%")
        print(f"  REAL:         {result['probabilities']['real']*100:.2f}%")
        print("="*70)
        
        print("\n📊 INTERPRETATION:")
        if result['certainty'] == 'LOW':
            print("  ⚠️  Model is UNCERTAIN about this image")
            print("     The probabilities are very close (margin < 0.1)")
            print("     This image may be borderline or unusual")
        elif result['margin'] < 0.05:
            print("  ⚠️  Model is BARELY making a decision (margin < 0.05)")
            print("     Consider this prediction unreliable")
        elif result['is_ai_generated'] and result['confidence'] > 0.8:
            print("  🤖 Model STRONGLY believes this is AI-generated")
            print("     Confidence is high")
        elif not result['is_ai_generated'] and result['confidence'] > 0.8:
            print("  📷 Model STRONGLY believes this is a real photo")
            print("     Confidence is high")
        
        print("\n💡 NOTE:")
        print("  Your model was trained on the CIFAKE dataset.")
        print("  If your real image is:")
        print("    - A phone camera photo")
        print("    - Low quality/compressed")
        print("    - Different style than training data")
        print("  The model may misclassify it as AI-generated.")
    else:
        print(f"❌ Path not found: {path}")