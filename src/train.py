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
from model import HydroTransNet, count_parameters, log_model_to_mlflow

TARGET_PARAMS = ['TSS', 'Turbidity', 'Chlorophyll']

class WaterQualityDataset(Dataset):
    def __init__(self, features, labels, seq_len=10):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.seq_len = seq_len
    def __len__(self):
        return max(0, len(self.labels) - self.seq_len + 1)
    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_len]
        y = self.labels[idx + self.seq_len - 1]
        return x, y

def load_data(features_file, labels_file):
    features_df = pd.read_csv(features_file)
    labels_df = pd.read_csv(labels_file)
    # Preprocessing (date removed, columns already scaled!)
    if 'date' in features_df.columns:
        features_df = features_df.drop('date', axis=1)
    if 'date' in labels_df.columns:
        labels_df = labels_df.drop('date', axis=1)
    X = features_df.values
    y = labels_df[TARGET_PARAMS].values
    print("features shape:", X.shape)
    print("labels shape:", y.shape)
    return X, y

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    batches = 0
    for batch_x, batch_y in tqdm(dataloader, desc="Training", leave=False):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_x = batch_x.transpose(0, 1)  # [seq, batch, features]
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        batches += 1
    return total_loss / batches if batches > 0 else None

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    batches = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_x = batch_x.transpose(0, 1)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            batches += 1
    return total_loss / batches if batches > 0 else None

def main(config_file='params.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    proc_features_file = os.path.join(PROCESSED_DATA_DIR, 'features_for_training.csv')
    proc_labels_file = os.path.join(PROCESSED_DATA_DIR, 'labels_for_training.csv')

    X, y = load_data(proc_features_file, proc_labels_file)
    n_samples = len(X)
    seq_len = config['model'].get('seq_len', 5)
    output_dim = config['model'].get('output_dim', y.shape[1])
    test_size = config['data']['test_size']
    split_idx = int(n_samples * (1 - test_size))
    if split_idx < seq_len:
        split_idx = seq_len
    if (n_samples - split_idx) < seq_len:
        if n_samples > seq_len:
            split_idx = n_samples - seq_len
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    print(f"Total samples: {n_samples}")
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    train_dataset = WaterQualityDataset(X_train, y_train, seq_len=seq_len)
    val_dataset = WaterQualityDataset(X_val, y_val, seq_len=seq_len)
    print(f"Train Dataset usable sequences: {len(train_dataset)}")
    print(f"Val Dataset usable sequences: {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("ERROR: Not enough training samples for sequence length. Aborting.")
        sys.exit(1)
    if len(val_dataset) == 0:
        print("WARNING: Validation set empty (seq_len or split too large). Training will proceed without validation.")

    train_loader = DataLoader(train_dataset, batch_size=config['model']['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['model']['batch_size'], shuffle=False, num_workers=0) if len(val_dataset) > 0 else None

    print("\nInitializing model...")
    model = HydroTransNet(
        input_dim=X.shape[1],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        output_dim=output_dim
    ).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['model']['learning_rate'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print("\nStarting training...")
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    model_path = os.path.join(MODELS_DIR, 'best_model.pt')

    for epoch in range(config['model']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['model']['epochs']}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = None
        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device)
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < current_lr:
                print(f"Learning rate reduced: {current_lr:.6f} -> {new_lr:.6f}")
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(f"Train Loss: {train_loss:.6f}", end="")
        if val_loss is not None:
            print(f", Val Loss: {val_loss:.6f}")
        else:
            print(", Val Loss: N/A (no validation set)")
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, model_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.6f})")
    history_path = os.path.join(MODELS_DIR, 'training_history.pkl')
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"\n✓ Training history saved at {history_path}")

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'params.yaml'
    if not os.path.exists(config_file):
        print(f"Error: Config file '{config_file}' does not exist.")
        sys.exit(1)
    main(config_file)
