import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import sys
import os
from src.config import *

def preprocess_data(input_file, output_file, scaler_file):
    """
    Clean, transform, and engineer features
    """
    input_path = os.path.join(RAW_DATA_DIR, input_file)
    df = pd.read_csv(input_path)
    
    print("Preprocessing data...")
    
    # 1. Handle missing values
    df = df.dropna()
    
    # 2. Remove duplicates
    df = df.drop_duplicates(subset=['date'])
    
    # 3. Feature engineering
    df['NDVI'] = (df['B8_NIR'] - df['B4_Red']) / (df['B8_NIR'] + df['B4_Red'] + 1e-8)
    df['NDWI'] = (df['B3_Green'] - df['B8_NIR']) / (df['B3_Green'] + df['B8_NIR'] + 1e-8)
    df['Turbidity_Index'] = df['B4_Red'] / (df['B3_Green'] + 1e-8)
    
    # 4. Select features (names must match dataset columns exactly)
    feature_cols = ['B2_Blue', 'B3_Green', 'B4_Red', 'B8_NIR', 'NDVI', 'NDWI', 'Turbidity_Index']
    
    X = df[feature_cols].values

    # 5. Normalize features using MinMaxScaler for [0, 1] or choose StandardScaler if you wish
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler for later inverse_transform
    scaler_path = os.path.join(MODELS_DIR, scaler_file)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
    
    # Create processed DataFrame
    df_processed = pd.DataFrame(X_scaled, columns=feature_cols)
    df_processed['date'] = df['date'].values
    
    # Save processed data
    output_path = os.path.join(PROCESSED_DATA_DIR, output_file)
    df_processed.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    print(f"Total records after preprocessing: {len(df_processed)}")
    
    return df_processed

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'sentinel2_validated.csv'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'processed_features.csv'
    scaler_file = sys.argv[3] if len(sys.argv) > 3 else 'scaler.pkl'
    
    preprocess_data(input_file, output_file, scaler_file)
