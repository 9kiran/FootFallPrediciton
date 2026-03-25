import requests
import pandas as pd

def fetch_precipitation_data(latitude=18.5898, longitude=73.7997, start_date='2024-01-01', end_date='2025-12-31'):
    """
    Fetch historical precipitation data from Open-Meteo API for a given location and date range.
    
    Parameters:
    - latitude: Location latitude (default: Hinjewadi, Pune = 18.5898)
    - longitude: Location longitude (default: Hinjewadi, Pune = 73.7997)
    - start_date: Start date in format 'YYYY-MM-DD'
    - end_date: End date in format 'YYYY-MM-DD'
    
    Returns:
    - DataFrame with columns: date, precipitation_mm
    """
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum",
        "timezone": "Asia/Kolkata"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Extract precipitation data
        dates = data['daily']['time']
        precipitation = data['daily']['precipitation_sum']
        
        # Create DataFrame
        df_rain = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'precipitation_mm': precipitation
        })
        
        print(f"✓ Successfully fetched precipitation data")
        print(f"  Location: Hinjewadi, Pune ({latitude}°N, {longitude}°E)")
        print(f"  Date range: {start_date} to {end_date}")
        print(f"  Total records: {len(df_rain)}")
        print(f"  Precipitation range: {df_rain['precipitation_mm'].min():.1f}mm - {df_rain['precipitation_mm'].max():.1f}mm")
        
        return df_rain
    
    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching data: {e}")
        return None

# Adding rain category based on precipitation_mm
def categorize_rain(mm):
    if pd.isna(mm):
        return 'Unknown'
    elif mm == 0:
        return 'No Rain'
    elif mm < 20:
        return 'Light Rain'
    elif mm < 45:
        return 'Moderate Rain'
    else:
        return 'Heavy Rain'

# define Pune holidays for 2024 and 2025 and mark dataframe accordingly
# Pune office holidays for 2024 and 2025
pune_holidays_2024 = [
    '2024-01-01', # New Year
    '2024-01-26', # Republic Day
    '2024-03-25', # Holi
    '2024-03-29', # Good Friday
    '2024-04-09', # Gudi Padava
    '2024-05-01', # Maharashtra Day
    '2024-08-15', # Independence Day
    '2024-09-17', # Anant Chaturdashi
    '2024-10-02', # Gandhi Jayanti
    '2024-11-01', # Diwali - Laxmi Pujan
    '2024-12-25', # Christmas
]

pune_holidays_2025 = [
    '2025-01-01', # New Year
    '2025-01-26', # Republic Day
    '2025-03-14', # Holi
    '2025-03-31', # Ramzan ID
    '2025-04-18', # Good Friday
    '2025-05-01', # Maharashtra Day
    '2025-08-15', # Independence Day
    '2025-08-27', # Anant Chaturdashi
    '2025-10-02', # Gandhi Jayanti
    '2025-10-21', # Diwali - Laxmi Pujan
    '2025-10-22', # Diwali - Padwa
    '2025-12-25', # Christmas
]
