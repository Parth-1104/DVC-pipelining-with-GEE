"""
Train the HydroTransNet model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
import yaml
import sys
import os
from tqdm import tqdm
from config import *
from model import HydroTransNet, count_parameters


class WaterQualityDataset(Dataset):
    """PyTorch Dataset for water quality data"""
    
    def __init__(self, features, labels, seq_len=10):
        """
        Args:
            features: numpy array of shape [n_samples, n_features]
            labels: numpy array of shape [n_samples, n_targets]
            seq_len: sequence length for temporal modeling
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.features) - self.seq_len + 1
    
    def __getitem__(self, idx):
        """Return a sequence and its target"""
        x = self.features[idx:idx + self.seq_len]
        y = self.labels[idx + self.seq_len - 1]
        return x, y


def load_data(features_file, labels_file):
    """Load preprocessed features and labels"""
    features_path = os.path.join(PROCESSED_DATA_DIR, features_file)
    labels_path = os.path.join(PROCESSED_DATA_DIR, labels_file)
    
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)
    
    # Drop date column if present
    if 'date' in features_df.columns:
        features_df = features_df.drop('date', axis=1)
    if 'date' in labels_df.columns:
        labels_df = labels_df.drop('date', axis=1)
    
    X = features_df.values
    y = labels_df.values
    
    return X, y


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in tqdm(dataloader, desc="Training"):
        # Move to device
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # Reshape: [batch, seq, features] -> [seq, batch, features]
        batch_x = batch_x.transpose(0, 1)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_x = batch_x.transpose(0, 1)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main(config_file='params.yaml'):
    """Main training function"""
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    X, y = load_data('processed_features.csv', 'labels.csv')
    
    # Split into train/val
    split_idx = int(len(X) * (1 - config['data']['test_size']))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    
    # Create datasets and dataloaders
    seq_len = config['model'].get('seq_len', 10)
    train_dataset = WaterQualityDataset(X_train, y_train, seq_len=seq_len)
    val_dataset = WaterQualityDataset(X_val, y_val, seq_len=seq_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['model']['batch_size'],
        shuffle=True,
        num_workers=0  # Changed from 2 to 0 for CPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['model']['batch_size'],
        shuffle=False,
        num_workers=0  # Changed from 2 to 0 for CPU
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = HydroTransNet(
        input_dim=X.shape[1],
        d_model=config['model'].get('d_model', 128),
        nhead=config['model'].get('nhead', 8),
        num_encoder_layers=config['model'].get('num_encoder_layers', 4),
        dim_feedforward=config['model'].get('dim_feedforward', 512),
        dropout=config['model'].get('dropout', 0.1),
        output_dim=y.shape[1]
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['model']['learning_rate'],
        weight_decay=1e-5
    )
    
    # Learning rate scheduler (FIXED - removed verbose parameter)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    patience = 0
    max_patience = 15
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(config['model']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['model']['epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Manual verbose output for LR change
        if new_lr < current_lr:
            print(f"Learning rate reduced: {current_lr:.6f} -> {new_lr:.6f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            
            model_path = os.path.join(MODELS_DIR, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, model_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.6f})")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Save training history
    history_path = os.path.join(MODELS_DIR, 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    print(f"\n✓ Training complete! Best val loss: {best_val_loss:.6f}")
    print(f"✓ Model saved to: {model_path}")



if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'params.yaml'
    main(config_file)
