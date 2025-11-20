"""
Validate raw data quality and log results to MLflow/DAGsHub
"""
import pandas as pd
import numpy as np
import sys
import os
import json
import mlflow
import dagshub
from config import *
import dagshub

# Initialize DAGsHub/MLflow

dagshub.init(repo_owner='Parth-1104', repo_name='DVC-pipelining-with-GEE', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Parth-1104/DVC-pipelining-with-GEE.mlflow")
mlflow.set_experiment("WaterQuality_Transformer")

def validate_data(input_file, output_file):
    """
    Validate and log data quality issues
    """
    input_path = os.path.join(RAW_DATA_DIR, input_file)
    df = pd.read_csv(input_path)

    print("=" * 50)
    print("DATA VALIDATION REPORT")
    print("=" * 50)

    # Check for missing values
    print("\n1. Missing Values:")
    missing = df.isnull().sum()
    print(missing[missing > 0])

    # Check for outliers (reflectance should be 0-1)
    print("\n2. Reflectance Range Check:")
    for band in BANDS:
        col_name = f"{band}_{['Blue', 'Green', 'Red', 'NIR'][BANDS.index(band)]}"
        if col_name in df.columns:
            min_val = df[col_name].min()
            max_val = df[col_name].max()
            print(f"{col_name}: [{min_val:.4f}, {max_val:.4f}]")
            if min_val < 0 or max_val > 1:
                print(f"  ⚠️  WARNING: Values outside expected range [0, 1]")

    # Check date format
    print("\n3. Date Format Check:")
    try:
        df['date'] = pd.to_datetime(df['date'])
        print(f"  ✓ Date range: {df['date'].min()} to {df['date'].max()}")
    except:
        print("  ✗ Invalid date format detected")

    # Check for duplicates
    print("\n4. Duplicate Check:")
    duplicates = df.duplicated(subset=['date']).sum()
    print(f"  Duplicate dates: {duplicates}")

    # Prepare validation report
    validation_report = {
        'total_records': len(df),
        'missing_values': missing.to_dict(),
        'date_range': f"{df['date'].min()} to {df['date'].max()}",
        'duplicates': int(duplicates)
    }

    # Save validated data
    output_path = os.path.join(RAW_DATA_DIR, output_file)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Validated data saved to {output_path}")

    # Save detailed report as JSON artifact
    report_path = os.path.join(RAW_DATA_DIR, 'validation_report.json')
    with open(report_path, "w") as f:
        json.dump(validation_report, f, indent=2)

    total_missing = sum(v for v in missing.to_dict().values() if v > 0)
    
    # Log to MLflow with stage tag
    with mlflow.start_run(run_name="stage_01_data_validation"):
        mlflow.set_tag("pipeline_stage", "data_validation")
        mlflow.set_tag("stage_order", "01")
        mlflow.log_param("input_file", input_file)
        mlflow.log_param("output_file", output_file)
        mlflow.log_param("bands", ",".join(BANDS))
        mlflow.log_metric("total_records", validation_report['total_records'])
        mlflow.log_metric("total_missing_values", total_missing)
        mlflow.log_metric("duplicates", validation_report['duplicates'])
        for col, cnt in validation_report['missing_values'].items():
            mlflow.log_metric(f"missing_{col}", cnt)
        mlflow.log_artifact(report_path)
        mlflow.log_artifact(output_path)

    print(f"✓ Validation metrics and report logged to MLflow/DAGsHub.")

    return validation_report


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'sentinel2_raw.csv'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'sentinel2_validated.csv'

    validate_data(input_file, output_file)
