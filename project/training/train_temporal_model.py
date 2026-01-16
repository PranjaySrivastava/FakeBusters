import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import random

# ========================================
# CONFIGURATION
# ========================================
data_dir = 'data/images_balanced'  # Using same images but as sequences
batch_size = 8  # Lower because we're loading sequences
num_epochs = 10
learning_rate = 0.0001
sequence_length = 16  # Number of frames per sequence

# ========================================
# CUSTOM DATASET FOR SEQUENCES
# ========================================
class VideoSequenceDataset(Dataset):
    """
    Creates sequences of frames from image folders.
    Each sequence is 'sequence_length' consecutive frames.
    """
    def __init__(self, data_dir, sequence_length=16, transform=None):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Get all images from real and fake folders
        self.sequences = []
        
        # Process real videos (label = 1)
        real_path = os.path.join(data_dir, 'real')
        if os.path.exists(real_path):
            real_files = sorted([f for f in os.listdir(real_path) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            # Group into sequences
            for i in range(0, len(real_files) - sequence_length, sequence_length // 2):
                seq = [os.path.join(real_path, real_files[i+j]) 
                       for j in range(sequence_length)]
                self.sequences.append((seq, 1))  # 1 = real
        
        # Process fake videos (label = 0)
        fake_path = os.path.join(data_dir, 'fake')
        if os.path.exists(fake_path):
            fake_files = sorted([f for f in os.listdir(fake_path) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            # Group into sequences
            for i in range(0, len(fake_files) - sequence_length, sequence_length // 2):
                seq = [os.path.join(fake_path, fake_files[i+j]) 
                       for j in range(sequence_length)]
                self.sequences.append((seq, 0))  # 0 = fake
        
        print(f"Created {len(self.sequences)} sequences ({sequence_length} frames each)")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence_paths, label = self.sequences[idx]
        
        # Load all frames in the sequence
        frames = []
        for path in sequence_paths:
            try:
                image = Image.open(path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                frames.append(image)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                # If error, use a blank frame
                frames.append(torch.zeros(3, 224, 224))
        
        # Stack frames: (sequence_length, 3, 224, 224)
        frames_tensor = torch.stack(frames)
        
        return frames_tensor, label

# ========================================
# TEMPORAL MODEL ARCHITECTURE
# ========================================
class TemporalDeepfakeDetector(nn.Module):
    """
    Uses CNN to extract features from each frame,
    then LSTM to analyze temporal patterns across frames.
    """
    def __init__(self, sequence_length=16):
        super(TemporalDeepfakeDetector, self).__init__()
        
        # CNN backbone for feature extraction (per frame)
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        # Remove the final FC layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze early layers
        for param in list(self.feature_extractor.parameters())[:-20]:
            param.requires_grad = False
        
        # LSTM to process temporal sequence
        self.lstm = nn.LSTM(
            input_size=512,  # ResNet18 output features
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # 2 classes: fake, real
        )
    
    def forward(self, x):
        # x shape: (batch, sequence_length, 3, 224, 224)
        batch_size, seq_len, c, h, w = x.shape
        
        # Process each frame through CNN
        # Reshape to (batch * sequence_length, 3, 224, 224)
        x = x.view(batch_size * seq_len, c, h, w)
        
        # Extract features
        features = self.feature_extractor(x)  # (batch * seq_len, 512, 1, 1)
        features = features.view(batch_size, seq_len, -1)  # (batch, seq_len, 512)
        
        # Process sequence through LSTM
        lstm_out, _ = self.lstm(features)  # (batch, seq_len, 256)
        
        # Use the last output for classification
        final_features = lstm_out[:, -1, :]  # (batch, 256)
        
        # Classify
        output = self.classifier(final_features)  # (batch, 2)
        
        return output

# ========================================
# DATA PREPARATION
# ========================================
def setup_data():
    print("Setting up data transformations...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("Loading sequence dataset...")
    full_dataset = VideoSequenceDataset(data_dir, sequence_length, transform)
    
    # Split into train (80%) and validation (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    }
    
    dataset_sizes = {'train': train_size, 'val': val_size}
    
    print(f"\n{'='*60}")
    print(f"Dataset loaded successfully!")
    print(f"{'='*60}")
    print(f"Training sequences:   {dataset_sizes['train']}")
    print(f"Validation sequences: {dataset_sizes['val']}")
    print(f"Frames per sequence:  {sequence_length}")
    print(f"{'='*60}\n")
    
    return dataloaders, dataset_sizes

# ========================================
# TRAINING FUNCTION
# ========================================
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs):
    best_acc = 0.0
    
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
            
            for sequences, labels in dataloaders[phase]:
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(sequences)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * sequences.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                batch_count += 1
                
                if batch_count % 5 == 0:
                    current_acc = running_corrects.double() / (batch_count * batch_size)
                    print(f'  {phase} [{batch_count}/{total_batches}] Loss: {loss.item():.4f} Acc: {current_acc:.4f}', end='\r')
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase:5s} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} ({running_corrects}/{dataset_sizes[phase]})')
            
            if phase == 'val':
                scheduler.step(epoch_loss)
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    os.makedirs('models', exist_ok=True)
                    torch.save(model.state_dict(), 'models/temporal_model.pth')
                    print(f'  ✓ Saved best model! (Acc: {best_acc:.4f})')
        
        print()
    
    print(f"\n{'='*60}")
    print('Training Complete!')
    print(f"{'='*60}")
    print(f'Best Validation Accuracy: {best_acc:.4f}')
    print(f"{'='*60}\n")
    
    return model

# ========================================
# MAIN EXECUTION
# ========================================
if __name__ == '__main__':
    print(f"Temporal Model Configuration:")
    print(f"  Sequence length: {sequence_length} frames")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print()
    
    try:
        # Setup data
        dataloaders, dataset_sizes = setup_data()
        
        # Setup model
        print("Building temporal model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TemporalDeepfakeDetector(sequence_length=sequence_length)
        model = model.to(device)
        print(f"✓ Using device: {device}")
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Trainable parameters: {trainable:,}\n")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        # Train
        model = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs)
        
        print("✅ Training completed successfully!")
        print(f"\n📁 Model saved to: models/temporal_model.pth")
        print(f"\n🚀 Next step: Create video inference script")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()