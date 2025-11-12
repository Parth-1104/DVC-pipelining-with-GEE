from fastapi import FastAPI
from pydantic import BaseModel
import torch
import pandas as pd
import yaml
import os
from src.model import HydroTransNet
from src.fetch_data import fetch_sentinel2_timeseries
from src.preprocess import preprocess_data

app = FastAPI()
model = None
config = None

# Load config once at startup
@app.on_event("startup")
async def startup_event():
    global model, config
    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model = HydroTransNet(
        input_dim=7,
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        output_dim=3
    )
    checkpoint_path = "models/trained_models/best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()


class PredictionRequest(BaseModel):
    coordinates: list  # polygon coords [[lng, lat], ...]
    start_date: str    # e.g. "2024-01-01"
    end_date: str      # e.g. "2024-01-10"
    input_file: str = 'sentinel2_validated.csv'
    output_file: str = 'processed_features.csv'
    scaler_file: str = 'scaler.pkl'


@app.get("/")
async def root():
    return {"message": "Water Quality Prediction API is running."}


@app.post("/predict")
async def predict(request: PredictionRequest):
    global model, config

    df_raw = fetch_sentinel2_timeseries(request.start_date, request.end_date, request.coordinates)
    if df_raw.empty:
        return {"error": "No Sentinel-2 data found for given coordinates and date range."}

    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, request.input_file)
    df_raw.to_csv(raw_path, index=False)

    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    try:
        preprocess_data(request.input_file, request.output_file, request.scaler_file)
    except Exception as e:
        return {"error": f"Preprocessing failed: {str(e)}"}

    processed_path = os.path.join(processed_dir, request.output_file)
    if not os.path.exists(processed_path):
        return {"error": "Processed data file not found after preprocessing."}

    df_processed = pd.read_csv(processed_path)
    if df_processed.empty:
        return {"error": "Processed data is empty after preprocessing."}

    seq_len = config['model']['seq_len']
    if len(df_processed) < seq_len:
        return {"error": f"Insufficient processed data length {len(df_processed)} for sequence length {seq_len}"}

    try:
        X = torch.tensor(df_processed.drop(columns=['date'], errors='ignore').values, dtype=torch.float32)
    except Exception as e:
        return {"error": f"Error preparing features for prediction: {str(e)}"}

    preds = []
    for i in range(len(X) - seq_len + 1):
        seq = X[i:i+seq_len].unsqueeze(1)
        with torch.no_grad():
            outputs = model(seq)
        preds.append(outputs.numpy().flatten().tolist())

    # Include date in output
    cols = ['TSS', 'Turbidity', 'pH']
    pred_df = pd.DataFrame(preds, columns=cols)
    pred_df['date'] = df_processed['date'].values[seq_len - 1:]  # align dates

    pred_dicts = pred_df.to_dict(orient='records')
    return {"predictions": pred_dicts}
    
class QuickPredictRequest(BaseModel):
    processed_features: list
@app.post("/predict_quick")
async def predict_quick(request: QuickPredictRequest):
    global model, config

    # Convert input features list to DataFrame
    df = pd.DataFrame(request.processed_features)
    if 'date' in df.columns:
        dates = df['date']
        df = df.drop(columns=['date'])
    else:
        dates = pd.Series(range(len(df)))

    seq_len = config['model']['seq_len']
    if len(df) < seq_len:
        return {"error": f"Insufficient data length {len(df)} for sequence length {seq_len}"}

    try:
        X = torch.tensor(df.values, dtype=torch.float32)
    except Exception as e:
        return {"error": f"Error preparing features for prediction: {str(e)}"}

    preds = []
    for i in range(len(X) - seq_len + 1):
        seq = X[i:i+seq_len].unsqueeze(1)
        with torch.no_grad():
            outputs = model(seq)
        preds.append(outputs.numpy().flatten().tolist())

    cols = ['TSS', 'Turbidity', 'pH']
    pred_df = pd.DataFrame(preds, columns=cols)
    pred_df['date'] = dates[seq_len - 1:].values

    pred_dicts = pred_df.to_dict(orient='records')
    return {"predictions": pred_dicts}
