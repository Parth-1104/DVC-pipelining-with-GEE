"""
Fetch Sentinel-2 data from Google Earth Engine
"""
import ee
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
from src.config import *



PROJECT_ID = 'ee-singhparth427'  
ee.Initialize(project=PROJECT_ID)

def fetch_sentinel2_timeseries(start_date, end_date, roi_coords):
    """
    Fetch Sentinel-2 time series data for a region of interest
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        roi_coords: List of coordinates defining the ROI
    
    Returns:
        pandas DataFrame with spectral bands and dates
    """
   
    roi = ee.Geometry.Polygon(roi_coords)
    
   
    collection = ee.ImageCollection(SENTINEL_COLLECTION) \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_COVER)) \
        .select(BANDS)
    
    
    count = collection.size().getInfo()
    print(f"Found {count} images in the date range")
    
    if count == 0:
        print("No images found. Check your date range and location.")
        return pd.DataFrame()
    
    
    image_list = collection.toList(count)
    
    data_records = []
    
    for i in range(count):
        try:
            image = ee.Image(image_list.get(i))
            
            
            date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            
            
            stats = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=roi,
                scale=SCALE,
                maxPixels=1e9
            ).getInfo()
            
            
            record = {
                'date': date,
                'B2_Blue': stats.get('B2', None) / 10000 if stats.get('B2') else None,
                'B3_Green': stats.get('B3', None) / 10000 if stats.get('B3') else None,
                'B4_Red': stats.get('B4', None) / 10000 if stats.get('B4') else None,
                'B8_NIR': stats.get('B8', None) / 10000 if stats.get('B8') else None,
            }
            
            data_records.append(record)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{count} images")
                
        except Exception as e:
            print(f"Error processing image {i}: {str(e)}")
            continue
    
    df = pd.DataFrame(data_records)
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
