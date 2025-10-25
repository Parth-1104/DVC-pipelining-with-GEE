import pandas as pd
import numpy as np
import os
from config import PROCESSED_DATA_DIR

input_file = os.path.join(PROCESSED_DATA_DIR, 'processed_features.csv')
output_file = os.path.join(PROCESSED_DATA_DIR, 'labels.csv')

df = pd.read_csv(input_file)
np.random.seed(42)

# Generate mock target parameters for testing
df['TSS'] = np.abs(np.random.normal(50, 10, len(df)))
df['Turbidity'] = np.abs(np.random.normal(15, 2, len(df)))
df['Chlorophyll'] = np.abs(np.random.normal(8, 1, len(df)))

df[['date', 'TSS', 'Turbidity', 'Chlorophyll']].to_csv(output_file, index=False)
print(f"✓ Generated synthetic labels at {output_file}")
