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
import mlflow
import dagshub


def load_model(model_path):
    """Load the trained Transformer model"""
    checkpoint = torch.load(model_path, map_location='cpu')
    model_config = checkpoint.get('config', {}).get('model', {})
    
    model = HydroTransNet(
        input_dim=len(['B2_Blue', 'B3_Green', 'B4_Red', 'B8_NIR', 'NDVI', 'NDWI', 'Turbidity_Index']),
        d_model=model_config.get('d_model', 128),
        nhead=model_config.get('nhead', 4),
        num_encoder_layers=model_config.get('num_encoder_layers', 8),
        dim_feedforward=model_config.get('dim_feedforward', 512),
        dropout=model_config.get('dropout', 0.01),
        output_dim=3
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def predict_new_data(model, new_data_path, output_path):
    """Run inference on unseen processed data and include NDVI and NDWI in output"""
    df = pd.read_csv(new_data_path)

    if 'date' in df.columns:
        dates = df['date']
        df_nodate = df.drop(columns=['date'])
    else:
        dates = np.arange(len(df))
        df_nodate = df

    # Make sure NDVI and NDWI exist
    ndvi_vals = df['NDVI'].values
    ndwi_vals = df['NDWI'].values

    X_new = torch.tensor(df_nodate.values, dtype=torch.float32)
    seq_len = 3  # Should match training config

    preds = []
    ndvi_seq = []
    ndwi_seq = []
    for i in range(len(X_new) - seq_len + 1):
        seq = X_new[i:i + seq_len].unsqueeze(1)  # [seq_len, 1, features]
        with torch.no_grad():
            outputs = model(seq)
        preds.append(outputs.numpy().flatten())
        ndvi_seq.append(ndvi_vals[i + seq_len - 1])
        ndwi_seq.append(ndwi_vals[i + seq_len - 1])

    preds = np.array(preds)
    pred_df = pd.DataFrame(preds, columns=['TSS mg/L', 'Turbidity NTU', 'Chlorophyll mg/L'])

    # Add NDVI, NDWI, and date to outputs (aligned with predictions)
    pred_df['date'] = dates[seq_len - 1:].values
    pred_df['NDVI'] = ndvi_seq
    pred_df['NDWI'] = ndwi_seq

    pred_df.to_csv(output_path, index=False)
    print(f"✓ Saved predictions to {output_path}")

    with mlflow.start_run(run_name="predict"):
        mlflow.log_param("seq_len", seq_len)
        mlflow.log_metric("num_predictions", len(pred_df))
        mlflow.log_artifact(output_path)

    # Include NDVI and NDWI in the output JSON as well
    return pred_df.to_dict(orient='records')



def main():
    model_path = os.path.join(MODELS_DIR, 'best_model.pt')
    new_data_path = os.path.join(DATA_DIR, 'processed', 'processed_features.csv')
    # Corrected output path
    output_path = os.path.join(MODELS_DIR, 'new_predictions.csv')
    
    model = load_model(model_path)
    preds_json = predict_new_data(model, new_data_path, output_path)
    # Print or further handle preds_json if needed
    print(preds_json)


if __name__ == "__main__":
    main()
