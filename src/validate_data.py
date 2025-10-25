"""
Validate raw data quality
"""
import pandas as pd
import numpy as np
import sys
import os
from config import *


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
    
    # Save validation report
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
    
    return validation_report


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'sentinel2_raw.csv'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'sentinel2_validated.csv'
    
    validate_data(input_file, output_file)
