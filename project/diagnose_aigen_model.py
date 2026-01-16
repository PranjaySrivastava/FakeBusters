import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision import datasets
import os
import numpy as np

print("="*70)
print("AI-GEN MODEL DIAGNOSTICS")
print("="*70)

# ========================================
# 1. CHECK MODEL FILE
# ========================================
print("\n1. Checking model file...")
if os.path.exists('models/aigen_model.pth'):
    checkpoint = torch.load('models/aigen_model.pth', map_location='cpu', weights_only=False)
    print(f"   ✓ Model found")
    print(f"   ✓ Parameters: {sum(p.numel() for p in checkpoint.values()):,}")
else:
    print("   ❌ Model not found!")
    exit(1)

# ========================================
# 2. CHECK TRAINING DATA
# ========================================
print("\n2. Checking training data structure...")
data_dir = 'data/cifake_organized'

if not os.path.exists(data_dir):
    print(f"   ❌ Data directory not found: {data_dir}")
else:
    print(f"   ✓ Data directory exists")
    
    # Check class folders
    train_dir = os.path.join(data_dir, 'train')
    if os.path.exists(train_dir):
        classes = sorted(os.listdir(train_dir))
        print(f"   ✓ Train classes: {classes}")
        
        for cls in classes:
            cls_path = os.path.join(train_dir, cls)
            if os.path.isdir(cls_path):
                count = len([f for f in os.listdir(cls_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"      - {cls}: {count:,} images")

# ========================================
# 3. VERIFY CLASS MAPPING
# ========================================
print("\n3. Verifying class mapping...")
dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'))
class_to_idx = dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

print(f"   Class to Index mapping: {class_to_idx}")
print(f"   Index to Class mapping: {idx_to_class}")
print(f"\n   IMPORTANT:")
print(f"   - Class 0 = {idx_to_class[0]}")
print(f"   - Class 1 = {idx_to_class[1]}")

# ========================================
# 4. LOAD MODEL AND TEST ON VALIDATION
# ========================================
print("\n4. Testing model on validation set...")

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(checkpoint)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(
    os.path.join(data_dir, 'val'),
    transform=transform
)

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"   Validation samples: {len(val_dataset)}")

# Test accuracy on validation set
correct = 0
total = 0
class_correct = {0: 0, 1: 0}
class_total = {0: 0, 1: 0}

predictions_by_class = {0: [], 1: []}

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        for i in range(labels.size(0)):
            label = labels[i].item()
            pred = predicted[i].item()
            prob = probabilities[i][pred].item()
            
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1
            
            predictions_by_class[label].append(prob)

accuracy = 100 * correct / total
print(f"\n   Overall Validation Accuracy: {accuracy:.2f}%")

for cls_idx in [0, 1]:
    cls_name = idx_to_class[cls_idx]
    cls_acc = 100 * class_correct[cls_idx] / class_total[cls_idx] if class_total[cls_idx] > 0 else 0
    avg_conf = np.mean(predictions_by_class[cls_idx]) if predictions_by_class[cls_idx] else 0
    
    print(f"\n   Class '{cls_name}' (idx={cls_idx}):")
    print(f"      Accuracy: {cls_acc:.2f}% ({class_correct[cls_idx]}/{class_total[cls_idx]})")
    print(f"      Avg Confidence: {avg_conf*100:.2f}%")

# ========================================
# 5. DIAGNOSIS
# ========================================
print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)

if class_correct[1] < class_total[1] * 0.5:  # Class 1 = real
    print("⚠️  WARNING: Model performs poorly on REAL images!")
    print("   Possible causes:")
    print("   1. Training data imbalance")
    print("   2. Real images in training are too different from test images")
    print("   3. Model is overfitting to fake/AI-generated patterns")
    print("\n   Solutions:")
    print("   - Add more diverse real images to training")
    print("   - Use data augmentation")
    print("   - Try a different architecture or lower learning rate")

if accuracy < 70:
    print("⚠️  WARNING: Overall accuracy is low!")
    print("   Model may need more training or better data")

if accuracy >= 70 and class_correct[0] >= class_total[0] * 0.7 and class_correct[1] >= class_total[1] * 0.7:
    print("✓ Model appears to be working well on validation data!")
    print("  If it fails on your real images, they may be different from training data.")

print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
print("Run: python test_aigen_model_enhanced.py your_image.jpg")
print("="*70)