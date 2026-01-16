import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os

# ========================================
# CONFIGURATION
# ========================================
data_dir = 'data/images_balanced'
batch_size = 32
num_epochs = 15
learning_rate = 0.0001

# ========================================
# DATA PREPARATION
# ========================================
def setup_data():
    print("Setting up data transformations...")
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    print("Loading datasets...")
    
    if not os.path.exists(data_dir):
        print(f"\n❌ ERROR: Data directory not found!")
        print(f"   Looking for: {data_dir}")
        exit(1)
    
    if not os.path.exists(f'{data_dir}/real') or not os.path.exists(f'{data_dir}/fake'):
        print(f"\n❌ ERROR: Missing 'real' or 'fake' folders!")
        exit(1)
    
    # Load all data
    print(f"Loading images from: {data_dir}")
    full_dataset = datasets.ImageFolder(data_dir)
    
    # Split into train (80%) and validation (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create wrapper to apply transforms
    class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
            self.dataset = subset.dataset
        
        def __getitem__(self, idx):
            actual_idx = self.subset.indices[idx]
            path, target = self.dataset.samples[actual_idx]
            from PIL import Image
            sample = Image.open(path).convert('RGB')
            return self.transform(sample), target
        
        def __len__(self):
            return len(self.subset)
    
    train_dataset_transformed = TransformedDataset(train_dataset, data_transforms['train'])
    val_dataset_transformed = TransformedDataset(val_dataset, data_transforms['val'])
    
    # FIXED: num_workers=0 for Windows compatibility
    dataloaders = {
        'train': DataLoader(train_dataset_transformed, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset_transformed, batch_size=batch_size, shuffle=False, num_workers=0)
    }
    
    dataset_sizes = {'train': train_size, 'val': val_size}
    class_names = full_dataset.classes
    
    print(f"\n{'='*60}")
    print(f"Dataset loaded successfully!")
    print(f"{'='*60}")
    print(f"Training images:   {dataset_sizes['train']}")
    print(f"Validation images: {dataset_sizes['val']}")
    print(f"Classes: {class_names}")
    print(f"{'='*60}\n")
    
    return dataloaders, dataset_sizes, class_names

# ========================================
# MODEL SETUP
# ========================================
def setup_model():
    print("Building model...")
    
    # Use ResNet18 pretrained
    model = models.resnet18(weights='IMAGENET1K_V1')
    
    # Freeze early layers - only train later layers
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False
    
    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"✓ Using device: {device}")
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"✓ Trainable parameters: {trainable:,} / {total:,}")
    print()
    
    return model, device

# ========================================
# TRAINING FUNCTION
# ========================================
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs):
    best_acc = 0.0
    best_loss = float('inf')
    
    print(f"{'='*60}")
    print("Starting Training")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            batch_count = 0
            total_batches = len(dataloaders[phase])
            
            # Iterate over data
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
                
                batch_count += 1
                
                if batch_count % 10 == 0:
                    current_acc = running_corrects.double() / (batch_count * batch_size)
                    print(f'  {phase} [{batch_count}/{total_batches}] Loss: {loss.item():.4f} Acc: {current_acc:.4f}', end='\r')
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase:5s} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} ({running_corrects}/{dataset_sizes[phase]})')
            
            # Step scheduler on validation loss
            if phase == 'val':
                scheduler.step(epoch_loss)
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    
                    os.makedirs('models', exist_ok=True)
                    torch.save(model.state_dict(), 'models/image_model.pth')
                    print(f'  ✓ Saved best model! (Acc: {best_acc:.4f}, Loss: {best_loss:.4f})')
        
        print()
    
    print(f"\n{'='*60}")
    print('Training Complete!')
    print(f"{'='*60}")
    print(f'Best Validation Accuracy: {best_acc:.4f}')
    print(f'Best Validation Loss: {best_loss:.4f}')
    print(f"{'='*60}\n")
    
    return model

# ========================================
# MAIN EXECUTION
# ========================================
if __name__ == '__main__':
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print()
    
    try:
        # Setup data
        dataloaders, dataset_sizes, class_names = setup_data()
        
        # Setup model
        model, device = setup_model()
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=False)
        
        # Train
        model = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs)
        
        print("✅ Training completed successfully!")
        print(f"\n📁 Model saved to: models/image_model.pth")
        print(f"\n🚀 Next steps:")
        print(f"   1. Test the model: python training/test_image_model.py <image_path>")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Partial model may have been saved to models/image_model.pth")
        
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()