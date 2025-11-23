# pretrain.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import os
import sys

def preprocess_data(
    features_file, 
    labels_file, 
    processed_features_out, 
    processed_labels_out, 
    feature_scaler_path, 
    label_scaler_path
):
    print(f"Loading features from: {features_file}")
    features_df = pd.read_csv(features_file)
    print(f"Loading labels from: {labels_file}")
    labels_df = pd.read_csv(labels_file)
    feature_dates = features_df['date']
    label_dates = labels_df['date']
    features_df = features_df.drop(columns=['date'], errors='ignore')
    labels_df = labels_df.drop(columns=['date', 'B2_Blue', 'B3_Green', 'B4_Red', 'B8_NIR', 'NDVI', 'NDWI', 'Turbidity_Index'], errors='ignore')
    
    # Clip negatives - best to do for all 3 targets!
    for param in ['TSS', 'Turbidity', 'Chlorophyll']:
        if param in labels_df.columns:
            labels_df[param] = labels_df[param].clip(lower=0)
    features_df.fillna(features_df.mean(), inplace=True)
    labels_df.fillna(labels_df.mean(), inplace=True)
    
    # Feature scaling (StandardScaler is fine if your features are unbounded, else MinMaxScaler)
    feature_scaler = StandardScaler()
    features_scaled = feature_scaler.fit_transform(features_df.values)
    os.makedirs(os.path.dirname(feature_scaler_path), exist_ok=True)
    joblib.dump(feature_scaler, feature_scaler_path)
    processed_features_df = pd.DataFrame(features_scaled, columns=features_df.columns)
    processed_features_df['date'] = feature_dates.values
    
    os.makedirs(os.path.dirname(processed_features_out), exist_ok=True)
    processed_features_df.to_csv(processed_features_out, index=False)
    print(f"Processed features saved to: {processed_features_out}")
    
    # Label scaling (MinMaxScaler is better for positive targets)
    label_scaler = MinMaxScaler()
    labels_scaled = label_scaler.fit_transform(labels_df.values)
    os.makedirs(os.path.dirname(label_scaler_path), exist_ok=True)
    joblib.dump(label_scaler, label_scaler_path)
    processed_labels_df = pd.DataFrame(labels_scaled, columns=labels_df.columns)
    processed_labels_df['date'] = label_dates.values
    
    os.makedirs(os.path.dirname(processed_labels_out), exist_ok=True)
    processed_labels_df.to_csv(processed_labels_out, index=False)
    print(f"Processed labels saved to: {processed_labels_out}")
    print(f"Preprocessing complete: {len(processed_features_df)} rows processed.")

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python pretrain.py <input_features.csv> <input_labels.csv> <output_features.csv> <output_labels.csv> <feature_scaler.pkl> <label_scaler.pkl>")
        sys.exit(1)
    preprocess_data(*sys.argv[1:])
