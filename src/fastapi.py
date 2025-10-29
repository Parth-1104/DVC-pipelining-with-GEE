from fastapi import FastAPI
from pydantic import BaseModel
import torch
import pandas as pd
from src.model import HydroTransNet
from src.fetch_data import fetch_sentinel2_timeseries
from src.preprocess import preprocess_data
import os

app = FastAPI()

class PredictionRequest(BaseModel):
    coordinates: list  # e.g. [[lng, lat]]
    start_date: str    # e.g. "2024-01-01"
    end_date: str      # e.g. "2024-01-10"
    input_file: str = 'sentinel2_validated.csv'
    output_file: str = 'processed_features.csv'
    scaler_file: str = 'scaler.pkl'


@app.post("/predict")
async def predict(request: PredictionRequest):
    # Step 1: Fetch raw data from GEE with inputs
    df_raw = fetch_sentinel2_timeseries(request.start_date, request.end_date, request.coordinates)
    
    # Step 2: Save raw data temporarily (optional if preprocess expects file)
    raw_path = os.path.join("data/raw", request.input_file)
    df_raw.to_csv(raw_path, index=False)
    
    # Step 3: Preprocess data (using paths from request)
    df_processed = preprocess_data(request.input_file, request.output_file, request.scaler_file)
    
    # Load processed data for model input
    processed_path = os.path.join("data/processed", request.output_file)
    df_processed = pd.read_csv(processed_path)
    
    # Step 4: Load model with config
    model = HydroTransNet(
        input_dim=7,
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        output_dim=3
    )
    model.load_state_dict(torch.load("models/trained_models/best_model.pt")['model_state_dict'])
    model.eval()
    
    # Step 5: Prepare tensor and predict
    X = torch.tensor(df_processed.drop(columns=['date']).values, dtype=torch.float32)
    seq_len = 5
    preds = []
    for i in range(len(X) - seq_len + 1):
        seq = X[i:i+seq_len].unsqueeze(1)
        with torch.no_grad():
            outputs = model(seq)
        preds.append(outputs.numpy().flatten())
    
    return {"predictions": preds}
