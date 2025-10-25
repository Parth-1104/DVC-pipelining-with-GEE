"""
Configuration file for the water quality prediction pipeline
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'trained_models')

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# GEE Configuration
LAKE_COORDS = [
    [77.1100, 28.7500],
    [77.1150, 28.7500],
    [77.1150, 28.7550],
    [77.1100, 28.7550],
    [77.1100, 28.7500]
]

SENTINEL_COLLECTION = 'COPERNICUS/S2_SR'
BANDS = ['B2', 'B3', 'B4', 'B8']
SCALE = 10
MAX_CLOUD_COVER = 20

# Target parameters
TARGET_PARAMS = ['TSS', 'Turbidity', 'Chlorophyll']

# Model parameters (to be loaded from params.yaml)
RANDOM_STATE = 42
TEST_SIZE = 0.2
