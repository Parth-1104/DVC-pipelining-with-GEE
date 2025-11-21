import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

def preprocess_data(features_file, labels_file, processed_features_out, processed_labels_out, scaler_path):
    print(f"Loading features from: {features_file}")
    features_df = pd.read_csv(features_file)
    print(f"Loading labels from: {labels_file}")
    labels_df = pd.read_csv(labels_file)

    # Separate date columns for later alignment
    feature_dates = features_df['date']
    label_dates = labels_df['date']

    # Drop date and band columns from labels (only keep targets)
    features_df = features_df.drop(columns=['date'])
    labels_df = labels_df.drop(columns=['date', 'B2_Blue','B3_Green','B4_Red','B8_NIR','NDVI','NDWI','Turbidity_Index'], errors='ignore')

    # Clip negative Turbidity and Chlorophyll to zero as physical constraint
    labels_df['Turbidity'] = labels_df['Turbidity'].clip(lower=0)
    labels_df['Chlorophyll'] = labels_df['Chlorophyll'].clip(lower=0)

    # Impute missing values by column mean
    features_df.fillna(features_df.mean(), inplace=True)
    labels_df.fillna(labels_df.mean(), inplace=True)

    # Scale features with StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df.values)

    # Ensure output directories exist
    os.makedirs(os.path.dirname(processed_features_out), exist_ok=True)
    os.makedirs(os.path.dirname(processed_labels_out), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    # Save scaler for inference
    print(f"Saving scaler to: {scaler_path}")
    joblib.dump(scaler, scaler_path)

    # Save processed features (with date)
    processed_features_df = pd.DataFrame(features_scaled, columns=features_df.columns)
    processed_features_df['date'] = feature_dates.values
    print(f"Saving processed features to: {processed_features_out}")
    processed_features_df.to_csv(processed_features_out, index=False)

    # Save processed labels (with date)
    processed_labels_df = labels_df.copy()
    processed_labels_df['date'] = label_dates.values
    print(f"Saving processed labels to: {processed_labels_out}")
    processed_labels_df.to_csv(processed_labels_out, index=False)

    print(f"Preprocessing complete: {len(processed_features_df)} rows processed.")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python pretrain.py <input_features.csv> <input_labels.csv> <output_features.csv> <output_labels.csv> <scaler.pkl>")
        sys.exit(1)

    input_features = sys.argv[1]
    input_labels = sys.argv[2]
    output_features = sys.argv[3]
    output_labels = sys.argv[4]
    scaler_file = sys.argv[5]

    preprocess_data(input_features, input_labels, output_features, output_labels, scaler_file)
