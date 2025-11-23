from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
import pandas as pd
import yaml
import os
import requests
from src.model import HydroTransNet
from src.fetch_data import fetch_sentinel2_timeseries
from src.preprocess import preprocess_data
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use allowed origins in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
config = None

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

    # 1. Fetch raw time series from Sentinel-2 in requested time window
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

    # 2. Prepare feature tensor [all rows, all columns except date]
    try:
        X = torch.tensor(df_processed.drop(columns=['date'], errors='ignore').values, dtype=torch.float32)
    except Exception as e:
        return {"error": f"Error preparing features for prediction: {str(e)}"}

    # 3. Generate predictions for each sequence window
    preds = []
    for i in range(len(X) - seq_len + 1):
        seq = X[i:i+seq_len].unsqueeze(1)
        with torch.no_grad():
            outputs = model(seq)
        preds.append(outputs.numpy().flatten().tolist())

    cols = ['TSS mg/L', 'Turbidity NTU', 'Chlorophyll ug/L']
    pred_df = pd.DataFrame(preds, columns=cols)
    pred_df['date'] = df_processed['date'].values[seq_len - 1:]
    pred_df['NDVI'] = df_processed['NDVI'].values[seq_len - 1:]
    pred_df['NDWI'] = df_processed['NDWI'].values[seq_len - 1:]
    pred_dicts = pred_df.to_dict(orient='records')
    return {"predictions": pred_dicts}

class GeminiReportRequest(BaseModel):
    lake_name: str
    location: str
    area: float
    chart_data: list  # [{date:..., tss:..., turbidity:..., ...}, ...]
    start_date: str
    end_date: str

@app.post("/gemini_report")
async def gemini_report(request: GeminiReportRequest):
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyDB7nno6S22RVUo0PmrjT9t6UX5HwL4Y-Q')
    prompt = f"""
    Lake Name: {request.lake_name}
    Location: {request.location}
    Area: {request.area} ha
    Date Range: {request.start_date} to {request.end_date}
    Recent Data Snapshot (up to 7 recent days):
    {request.chart_data[-7:] if len(request.chart_data) >= 7 else request.chart_data}
    Please analyze the water quality trends and parameters,
    flag anomalies or risks, correlate parameters to possible pollution sources,
    and recommend actionable government interventions.
    Output a concise, helpful report for policymakers.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=50)
        data = response.json()
        candidates = data.get("candidates", [])
        if candidates and "content" in candidates[0] and "parts" in candidates[0]["content"] and candidates[0]["content"]["parts"]:
            report_text = candidates[0]["content"]["parts"][0]["text"]
            return {"report": report_text}
        else:
            err = data.get("error", {}).get("message", "No report generated (API issue or quota exceeded).")
            return {"error": err}
    except Exception as e:
        return {"error": str(e)}


@app.post("/currdate")
async def current_week_prediction(request: dict = {}):
    global model, config
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
    coordinates = request.get('coordinates', None)
    if not coordinates:
        return {"error": "Please provide 'coordinates' in the request body."}
    df_raw = fetch_sentinel2_timeseries(start_date, end_date, coordinates)
    if df_raw.empty:
        return {"error": f"No Sentinel-2 data found for coordinates in the last week: {start_date} to {end_date}"}
    try:
        df_processed = preprocess_data_inline(df_raw)
    except Exception as e:
        return {"error": f"Preprocessing failed: {str(e)}"}
    if df_processed.empty:
        return {"error": "Processed data is empty after preprocessing."}
    seq_len = config['model']['seq_len']
    if len(df_processed) < seq_len:
        return {"error": f"Insufficient data length {len(df_processed)} for sequence length {seq_len}"}
    try:
        X = torch.tensor(df_processed.drop(columns=['date'], errors='ignore').values, dtype=torch.float32)
    except Exception as e:
        return {"error": f"Error preparing features for prediction: {str(e)}"}
    with torch.no_grad():
        input_seq = X[-seq_len:].unsqueeze(1)
        output = model(input_seq).numpy().flatten()
    last_date = df_processed['date'].values[-1]
    ndvi_val = float(df_processed['NDVI'].values[-1])
    ndwi_val = float(df_processed['NDWI'].values[-1])
    result = {
        "TSS mg/L": float(output[0]),
        "Turbidity NTU": float(output[1]),
        "Chlorophyll ug/L": float(output[2]),
        "date": str(last_date),
        "NDVI": ndvi_val,
        "NDWI": ndwi_val
    }
    return result

def preprocess_data_inline(df):
    df = df.dropna().drop_duplicates(subset=['date'])
    df['NDVI'] = (df['B8_NIR'] - df['B4_Red']) / (df['B8_NIR'] + df['B4_Red'] + 1e-8)
    df['NDWI'] = (df['B3_Green'] - df['B8_NIR']) / (df['B3_Green'] + df['B8_NIR'] + 1e-8)
    df['Turbidity_Index'] = df['B4_Red'] / (df['B3_Green'] + 1e-8)
    feature_cols = ['B2_Blue', 'B3_Green', 'B4_Red', 'B8_NIR', 'NDVI', 'NDWI', 'Turbidity_Index']
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(df[feature_cols].values)
    processed_df = pd.DataFrame(features_scaled, columns=feature_cols)
    processed_df['date'] = df['date'].values
    processed_df['NDVI'] = df['NDVI'].values
    processed_df['NDWI'] = df['NDWI'].values
    return processed_df
