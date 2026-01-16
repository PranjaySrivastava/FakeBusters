import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os

# ========================================
# CONFIGURATION - OPTIMIZED
# ========================================
data_dir = 'data/cifake_organized'
batch_size = 32  # Reduced from 64
num_epochs = 10  # Reduced from 15 for faster training
learning_rate = 0.0001  # Lower learning rate

print("="*70)
print("AI-GENERATED IMAGE DETECTOR - Training (FIXED)")
print("="*70)

# ========================================
# DATA PREPARATION
# ========================================
print("\n1. Setting up data transformations...")

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("2. Loading dataset...")

if not os.path.exists(data_dir):
    print(f"\n❌ ERROR: {data_dir} not found!")
    print("   Please run organize_cifake_simple.py first")
    exit(1)

image_datasets = {
    'train': datasets.ImageFolder(f'{data_dir}/train', data_transforms['train']),
    'val': datasets.ImageFolder(f'{data_dir}/val', data_transforms['val'])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=0)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print(f"\n{'='*70}")
print(f"Dataset Info:")
print(f"{'='*70}")
print(f"Training images:   {dataset_sizes['train']:,}")
print(f"Validation images: {dataset_sizes['val']:,}")
print(f"Classes: {class_names}")
print(f"Class mapping: {image_datasets['train'].class_to_idx}")
print(f"{'='*70}\n")

# Verify class balance
print("Checking class balance...")
train_fake = len([f for f in os.listdir(f'{data_dir}/train/fake')])
train_real = len([f for f in os.listdir(f'{data_dir}/train/real')])
print(f"Train - Fake: {train_fake:,}, Real: {train_real:,}")

# ========================================
# MODEL SETUP - IMPROVED
# ========================================
print("\n3. Building model...")

model = models.resnet18(weights='IMAGENET1K_V1')  # Use new syntax

# Freeze early layers - only train last few
for name, param in model.named_parameters():
    if 'layer4' not in name and 'fc' not in name:
        param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"   ✓ Model: ResNet18")
print(f"   ✓ Device: {device}")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   ✓ Trainable params: {trainable:,}\n")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

# ========================================
# TRAINING
# ========================================
print("="*70)
print("Starting Training")
print("="*70)

best_acc = 0.0

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    print('-' * 70)
    
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        
        for images, labels in dataloaders[phase]:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        
        print(f'{phase:5s} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} ({running_corrects}/{dataset_sizes[phase]})')
        
        if phase == 'val':
            scheduler.step(epoch_loss)
            
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                os.makedirs('models', exist_ok=True)
                torch.save(model.state_dict(), 'models/aigen_model.pth')
                print(f'      ✓ Saved! (Best acc: {best_acc:.4f})')

print(f"\n{'='*70}")
print('Training Complete!')
print(f"{'='*70}")
print(f'Best Validation Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)')
print(f"Model saved: models/aigen_model.pth")
print(f"{'='*70}\n")