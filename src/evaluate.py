"""
Evaluate the trained HydroTransNet model
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yaml
import pickle
import os
import sys
from config import *
from model import HydroTransNet
from train import WaterQualityDataset, load_data


def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    metrics = {}
    
    # Overall metrics
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Per-parameter metrics
    param_names = TARGET_PARAMS
    for i, param in enumerate(param_names):
        metrics[f'{param}_rmse'] = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        metrics[f'{param}_mae'] = mean_absolute_error(y_true[:, i], y_pred[:, i])
        metrics[f'{param}_r2'] = r2_score(y_true[:, i], y_pred[:, i])
    
    return metrics


def plot_predictions(y_true, y_pred, save_path):
    """Plot predicted vs actual values"""
    param_names = TARGET_PARAMS
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, (ax, param) in enumerate(zip(axes, param_names)):
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5)
        ax.plot([y_true[:, i].min(), y_true[:, i].max()],
                [y_true[:, i].min(), y_true[:, i].max()],
                'r--', lw=2)
        ax.set_xlabel(f'Actual {param}')
        ax.set_ylabel(f'Predicted {param}')
        ax.set_title(f'{param} (R² = {r2_score(y_true[:, i], y_pred[:, i]):.3f})')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved prediction plot to {save_path}")
    plt.close()


def plot_residuals(y_true, y_pred, save_path):
    """Plot residuals"""
    param_names = TARGET_PARAMS
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, (ax, param) in enumerate(zip(axes, param_names)):
        ax.scatter(y_pred[:, i], residuals[:, i], alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel(f'Predicted {param}')
        ax.set_ylabel('Residual')
        ax.set_title(f'{param} Residuals')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved residual plot to {save_path}")
    plt.close()


def evaluate_model(model_path, config_file='params.yaml'):
    """Evaluate the trained model"""
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    X, y = load_data('processed_features.csv', 'labels.csv')
    
    # Use test split
    split_idx = int(len(X) * (1 - config['data']['test_size']))
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    print(f"Test samples: {len(X_test)}")
    

    seq_len = config['model'].get('seq_len', 10)
    test_dataset = WaterQualityDataset(X_test, y_test, seq_len=seq_len)
    
   
    print("\nLoading model...")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = HydroTransNet(
        input_dim=X.shape[1],
        d_model=checkpoint['config']['model'].get('d_model', 128),
        nhead=checkpoint['config']['model'].get('nhead', 8),
        num_encoder_layers=checkpoint['config']['model'].get('num_encoder_layers', 4),
        dim_feedforward=checkpoint['config']['model'].get('dim_feedforward', 512),
        dropout=checkpoint['config']['model'].get('dropout', 0.1),
        output_dim=y.shape[1]
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✓ Model loaded successfully")
    
 
    print("\nMaking predictions...")
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            x, y_true = test_dataset[i]
            x = x.unsqueeze(1).to(device)  
            
            y_pred = model(x)
            
            predictions.append(y_pred.cpu().numpy())
            actuals.append(y_true.numpy())
    
    y_pred = np.array(predictions)
    y_true = np.array(actuals)
    # Remove extra dimensions (e.g., [n, 1, 3] → [n, 3])
    if len(y_pred.shape) == 3:
        y_pred = y_pred.squeeze(1)

    if len(y_true.shape) == 3:
        y_true = y_true.squeeze(1)

    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(y_true, y_pred)
    
    # Print metrics
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(f"\nOverall Metrics:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    
    print(f"\nPer-Parameter Metrics:")
    for param in TARGET_PARAMS:
        print(f"\n{param}:")
        print(f"  RMSE: {metrics[f'{param}_rmse']:.4f}")
        print(f"  MAE:  {metrics[f'{param}_mae']:.4f}")
        print(f"  R²:   {metrics[f'{param}_r2']:.4f}")
    
    # Save metrics
    metrics_path = os.path.join(MODELS_DIR, 'evaluation_metrics.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"\n✓ Metrics saved to {metrics_path}")
    
    # Create visualizations
    print("\nGenerating plots...")
    plot_predictions(y_true, y_pred, os.path.join(MODELS_DIR, 'predictions.png'))
    plot_residuals(y_true, y_pred, os.path.join(MODELS_DIR, 'residuals.png'))
    
    # Save predictions
    predictions_df = pd.DataFrame(y_pred, columns=TARGET_PARAMS)
    predictions_df.to_csv(os.path.join(MODELS_DIR, 'test_predictions.csv'), index=False)
    
    print("\n✓ Evaluation complete!")
    
    return metrics


if __name__ == "__main__":
    model_path = os.path.join(MODELS_DIR, 'best_model.pt')
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'params.yaml'
    
    evaluate_model(model_path, config_file)
