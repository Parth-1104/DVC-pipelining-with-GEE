"""
Prediction script for the trained HydroTransNet model
Generates water quality predictions using processed new data.
"""

import torch
import pandas as pd
import numpy as np
import os
from config import *
from model import HydroTransNet
from train import WaterQualityDataset
import mlflow
import dagshub


def load_model(model_path):
    """Load the trained Transformer model"""
    checkpoint = torch.load(model_path, map_location='cpu')
    model_config = checkpoint.get('config', {}).get('model', {})
    
    model = HydroTransNet(
        input_dim=len(['B2_Blue', 'B3_Green', 'B4_Red', 'B8_NIR', 'NDVI', 'NDWI', 'Turbidity_Index']),
        d_model=model_config.get('d_model', 128),
        nhead=model_config.get('nhead', 8),
        num_encoder_layers=model_config.get('num_encoder_layers', 4),
        dim_feedforward=model_config.get('dim_feedforward', 512),
        dropout=model_config.get('dropout', 0.1),
        output_dim=3
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def predict_new_data(model, new_data_path, output_path):
    """Run inference on unseen processed data"""
    df = pd.read_csv(new_data_path)
    if 'date' in df.columns:
        dates = df['date']
        df = df.drop(columns=['date'])
    else:
        dates = np.arange(len(df))
    X_new = torch.tensor(df.values, dtype=torch.float32)
    seq_len = 5  # Should match training config

    preds = []
    for i in range(len(X_new) - seq_len + 1):
        seq = X_new[i:i + seq_len].unsqueeze(1)  # [seq_len, 1, features]
        with torch.no_grad():
            outputs = model(seq)
        preds.append(outputs.numpy().flatten())
    preds = np.array(preds)
    pred_df = pd.DataFrame(preds, columns=['TSS_pred', 'Turbidity_pred', 'Chlorophyll_pred'])
    pred_df['date'] = dates[seq_len - 1:].values  # align remaining timestamps
    pred_df.to_csv(output_path, index=False)
    print(f"✓ Saved predictions to {output_path}")

    with mlflow.start_run(run_name="predict"):
        mlflow.log_param("seq_len", seq_len)
        mlflow.log_metric("num_predictions", len(pred_df))
        mlflow.log_artifact(output_path)



def main():
    model_path = os.path.join(MODELS_DIR, 'best_model.pt')
    new_data_path = os.path.join(DATA_DIR, 'processed', 'processed_features.csv')
    
    # Corrected output path
    output_path = os.path.join(MODELS_DIR, 'new_predictions.csv')
    
    model = load_model(model_path)
    predict_new_data(model, new_data_path, output_path)



if __name__ == "__main__":
    main()
