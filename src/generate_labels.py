import ee
import pandas as pd
import os
from config import PROCESSED_DATA_DIR, LAKE_COORDS

# Authenticate and initialize EE
ee.Authenticate()
PROJECT_ID = 'ee-singhparth427'
ee.Initialize(project=PROJECT_ID)

# Input/output paths
features_file = os.path.join(PROCESSED_DATA_DIR, 'processed_features.csv')
output_file = os.path.join(PROCESSED_DATA_DIR, 'labels.csv')


features_df = pd.read_csv(features_file)
features_df['date'] = pd.to_datetime(features_df['date'])
features_df = features_df.sort_values('date')


roi = ee.Geometry.Polygon(LAKE_COORDS)


custom_start_date = '2019-01-01'   # Set as needed
custom_end_date = '2025-10-31'     # Set as needed
start_date = custom_start_date
end_date = custom_end_date
print(f"Using fixed date window: {start_date} to {end_date}")

landsat_col = (
    ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(roi)
    .filterDate(start_date, end_date)
    .filter(ee.Filter.lt('CLOUD_COVER', 60))
    .select(['SR_B3', 'SR_B4', 'SR_B5', 'SR_B6'])
)

# Optional: print available Landsat dates for diagnostics
scenes = landsat_col.aggregate_array('system:time_start').getInfo()
print(f"Found {len(scenes)} Landsat scenes in fixed date window:")
for ms in scenes[:20]:  # just first 10 for brevity
    print(ee.Date(ms).format('YYYY-MM-dd').getInfo())

def estimate_labels(image):
    tss = image.select('SR_B5').divide(image.select('SR_B4')).rename('TSS')
    turbidity = image.select('SR_B6').divide(image.select('SR_B5')).rename('Turbidity')
    chlorophyll = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('Chlorophyll')
    return image.addBands([tss, turbidity, chlorophyll]).set(
        'system:time_start', image.get('system:time_start')
    )

labeled_images = landsat_col.map(estimate_labels)

def get_image_info(image):
    date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
    means = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=100,
        maxPixels=1e9,
        bestEffort=True
    )
    return ee.Feature(None, {
        'date': date,
        'TSS': means.get('TSS'),
        'Turbidity': means.get('Turbidity'),
        'Chlorophyll': means.get('Chlorophyll')
    })

feature_collection = labeled_images.map(get_image_info)
features = feature_collection.getInfo()['features']

# Build a dataframe with all valid label dates/values
labels_list = []
for feat in features:
    props = feat['properties']
    if props['TSS'] is not None:
        labels_list.append({
            'date': props['date'],
            'TSS': props['TSS'],
            'Turbidity': props['Turbidity'],
            'Chlorophyll': props['Chlorophyll']
        })
labels_df = pd.DataFrame(labels_list)
labels_df['date'] = pd.to_datetime(labels_df['date'])
labels_df = labels_df.sort_values('date')

# Flexible merge: For each features_df row, attach nearest label within tolerance (e.g., ±7 days)
tolerance_days = 12
merged = pd.merge_asof(
    features_df,
    labels_df,
    on='date',
    direction='nearest',
    tolerance=pd.Timedelta(f'{tolerance_days} days')
)

# Drop samples with missing labels (still unmatched)
before_count = len(merged)
merged = merged.dropna(subset=['TSS', 'Turbidity', 'Chlorophyll'])
after_count = len(merged)
print(f"Merged features+labels: before dropna {before_count} ➔ after {after_count}")

MIN_LABELED_ROWS = 20  # adjust as needed for splitting
if after_count < MIN_LABELED_ROWS:
    print(
        f"WARNING: Only {after_count} labeled rows found! "
        f"At least {MIN_LABELED_ROWS} required for stable train/validation splits.\n"
        f"Consider increasing tolerance_days, date range, or checking your ROI."
    )
else:
    print(f"✓ Sufficient labeled data for split ({MIN_LABELED_ROWS}+)")

# Save the new dataset for downstream training/validation
merged.to_csv(output_file, index=False)
print(f"✓ Merged data with labels saved at {output_file}")
