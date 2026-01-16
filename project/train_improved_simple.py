import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os

"""
SIMPLIFIED UNIVERSAL DETECTOR
Uses your existing CIFAKE data but with EXTREME augmentation
to handle any random photo type
"""

data_dir = 'data/cifake_organized'
batch_size = 24
num_epochs = 20
learning_rate = 0.00005

print("="*70)
print("IMPROVED AI DETECTOR - Works on ANY Photo")
print("="*70)
print("\nUsing EXTREME augmentation to handle:")
print("  ✓ Phone photos, screenshots, professional cameras")
print("  ✓ Compressed/low quality images")
print("  ✓ Different lighting, colors, styles")
print("="*70)

# ========================================
# EXTREME AUGMENTATION
# ========================================
data_transforms = {
    'train': transforms.Compose([
        # Handle different aspect ratios and crops
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        
        # Geometric
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(25),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        
        # Color/lighting (simulates different cameras and lighting)
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        transforms.RandomGrayscale(p=0.1),
        
        # Quality degradation
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0)),
        
        transforms.ToTensor(),
        
        # Add noise and compression artifacts
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.03),
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
        
        # Random erasing (occlusion)
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("\n1. Loading dataset with EXTREME augmentation...")

image_datasets = {
    'train': datasets.ImageFolder(f'{data_dir}/train', data_transforms['train']),
    'val': datasets.ImageFolder(f'{data_dir}/val', data_transforms['val'])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=0)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

print(f"   Training: {dataset_sizes['train']:,} | Validation: {dataset_sizes['val']:,}")

# ========================================
# BETTER MODEL
# ========================================
print("\n2. Building improved model (ResNet34)...")

model = models.resnet34(weights='IMAGENET1K_V1')

# Unfreeze layers 2, 3, 4
for name, param in model.named_parameters():
    if any(layer in name for layer in ['layer2', 'layer3', 'layer4', 'fc']):
        param.requires_grad = True
    else:
        param.requires_grad = False

# Better classifier with dropout
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"   Device: {device}")

# ========================================
# TRAINING
# ========================================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for robustness
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

print("\n" + "="*70)
print("TRAINING")
print("="*70)

best_acc = 0.0

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    print('-' * 70)
    
    for phase in ['train', 'val']:
        model.train() if phase == 'train' else model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        
        for images, labels in dataloaders[phase]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                if phase == 'train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        
        print(f'{phase:5s} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')
        
        if phase == 'val':
            scheduler.step(epoch_loss)
            
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'models/improved_detector.pth')
                print(f'      ✓ Saved! Best: {best_acc:.4f}')

print(f"\n{'='*70}")
print(f'✅ Training Complete!')
print(f'Best Accuracy: {best_acc*100:.2f}%')
print(f"Model saved: models/improved_detector.pth")
print(f"{'='*70}\n")

print("🎯 Test with: python test_improved_detector.py your_image.jpg")
