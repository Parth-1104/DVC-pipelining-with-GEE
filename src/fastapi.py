from fastapi import FastAPI
from pydantic import BaseModel
import torch
import pandas as pd
import yaml
import os
from src.model import HydroTransNet
from src.fetch_data import fetch_sentinel2_timeseries
from src.preprocess import preprocess_data
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <-- change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    # Prepare features and get NDVI, NDWI values aligned with prediction windows
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

    cols = ['TSS mg/L', 'Turbidity NTU', 'Chlorophyll ug/L']
    pred_df = pd.DataFrame(preds, columns=cols)
    pred_df['date'] = df_processed['date'].values[seq_len - 1:]

    # Add NDVI and NDWI columns aligned with prediction timestamps
    pred_df['NDVI'] = df_processed['NDVI'].values[seq_len - 1:]
    pred_df['NDWI'] = df_processed['NDWI'].values[seq_len - 1:]

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

@app.post("/currdate")
async def current_week_prediction(request: dict = {}):
    """
    Get water quality prediction for the last week's data.
    """
    global model, config

    # Define date range: last week
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=20)).strftime('%Y-%m-%d')

    # Use default ROI coordinates or get from request if provided
    coordinates = request.get('coordinates', None)
    if not coordinates:
        return {"error": "Please provide 'coordinates' in the request body."}

    # Fetch raw Sentinel-2 data for last week
    df_raw = fetch_sentinel2_timeseries(start_date, end_date, coordinates)
    if df_raw.empty:
        return {"error": f"No Sentinel-2 data found for coordinates in the last week: {start_date} to {end_date}"}

    # Preprocess fetched data
    try:
        df_processed = preprocess_data_inline(df_raw)  # Use inline or helper preprocessing function returning DataFrame
    except Exception as e:
        return {"error": f"Preprocessing failed: {str(e)}"}

    if df_processed.empty:
        return {"error": "Processed data is empty after preprocessing."}

    seq_len = config['model']['seq_len']
    if len(df_processed) < seq_len:
        return {"error": f"Insufficient data length {len(df_processed)} for sequence length {seq_len}"}

    # Prepare input features tensor
    try:
        X = torch.tensor(df_processed.drop(columns=['date'], errors='ignore').values, dtype=torch.float32)
    except Exception as e:
        return {"error": f"Error preparing features for prediction: {str(e)}"}

    # Predict on last sequence
    with torch.no_grad():
        input_seq = X[-seq_len:].unsqueeze(1)  # [seq_len, batch=1, features]
        output = model(input_seq).numpy().flatten()

    # Extract last date, NDVI, NDWI aligned with prediction
    last_date = df_processed['date'].values[-1]
    ndvi_val = float(df_processed['NDVI'].values[-1])
    ndwi_val = float(df_processed['NDWI'].values[-1])

    # Construct response
    result = {
        "TSS mg/L": float(output[0]),
        "Turbidity NTU": float(output[1]),
        "Chlorophyll ug/L": float(output[2]),
        "date": str(last_date),
        "NDVI": ndvi_val,
        "NDWI": ndwi_val
    }
    return result


# Helper inline preprocessing, similar to your preprocess_data but accepting DataFrame directly.
def preprocess_data_inline(df):
    # 1. Drop NA and duplicates (optional)
    df = df.dropna().drop_duplicates(subset=['date'])

    # 2. Feature engineering
    df['NDVI'] = (df['B8_NIR'] - df['B4_Red']) / (df['B8_NIR'] + df['B4_Red'] + 1e-8)
    df['NDWI'] = (df['B3_Green'] - df['B8_NIR']) / (df['B3_Green'] + df['B8_NIR'] + 1e-8)
    df['Turbidity_Index'] = df['B4_Red'] / df['B3_Green']

    feature_cols = ['B2_Blue', 'B3_Green', 'B4_Red', 'B8_NIR', 'NDVI', 'NDWI', 'Turbidity_Index']

    # 3. Fill missing if any, scale features (using runtime StandardScaler)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df[feature_cols].values)

    processed_df = pd.DataFrame(features_scaled, columns=feature_cols)
    processed_df['date'] = df['date'].values
    # Add NDVI and NDWI columns as original scaled versions for response use
    processed_df['NDVI'] = df['NDVI'].values
    processed_df['NDWI'] = df['NDWI'].values

    return processed_df


