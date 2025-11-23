"""
Fetch Sentinel-2 data from Google Earth Engine
"""
import ee
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
from config import *



PROJECT_ID = 'ee-singhparth427'  
ee.Initialize(project=PROJECT_ID)

def fetch_sentinel2_timeseries(start_date, end_date, roi_coords):
    """
    Fetch Sentinel-2 time series data efficiently by server-side aggregation and one-shot download.
    """
    roi = ee.Geometry.Polygon(roi_coords)
    collection = (
        ee.ImageCollection(SENTINEL_COLLECTION)
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_COVER))
        .select(BANDS)
    )

    def image_to_feature(image):
        stats = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=SCALE,
            maxPixels=1e9
        )
        date = image.date().format('YYYY-MM-dd')
        return ee.Feature(None, {
            'date': date,
            'B2_Blue': stats.get('B2'),
            'B3_Green': stats.get('B3'),
            'B4_Red': stats.get('B4'),
            'B8_NIR': stats.get('B8'),
        })

    feature_collection = collection.map(image_to_feature)

    # One call to getFeatureCollection info (server side)
    data = feature_collection.getInfo()
    records = []
    for f in data['features']:
        props = f['properties']
        record = {
            'date': props['date'],
            'B2_Blue': props['B2_Blue'] / 10000 if props['B2_Blue'] else None,
            'B3_Green': props['B3_Green'] / 10000 if props['B3_Green'] else None,
            'B4_Red': props['B4_Red'] / 10000 if props['B4_Red'] else None,
            'B8_NIR': props['B8_NIR'] / 10000 if props['B8_NIR'] else None,
        }
        records.append(record)

    df = pd.DataFrame(records)
    return df



def main(start_date, end_date, output_file):
    """
    Main function to fetch and save data
    """
    print(f"Fetching Sentinel-2 data from {start_date} to {end_date}")
    
    df = fetch_sentinel2_timeseries(start_date, end_date, LAKE_COORDS)
    
    if not df.empty:
        # Save to CSV
        output_path = os.path.join(RAW_DATA_DIR, output_file)
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
        print(f"Total records: {len(df)}")
    else:
        print("No data fetched")


if __name__ == "__main__":
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=20)).strftime('%Y-%m-%d')
    
    if len(sys.argv) > 1:
        start_date = sys.argv[1]
    if len(sys.argv) > 2:
        end_date = sys.argv[2]
    
    main(start_date, end_date, 'sentinel2_raw.csv')
