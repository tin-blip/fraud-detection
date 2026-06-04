"""Data preprocessing utilities"""

import pandas as pd
import ipaddress

def ip_to_int(ip_string):
    """Convert IP address to integer"""
    try:
        return int(ipaddress.IPv4Address(str(ip_string)))
    except:
        return None

def load_and_clean_data(filepath):
    """Load and clean fraud data"""
    df = pd.read_csv(filepath)
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    return df

def create_time_features(df):
    """Create time-based features"""
    df['time_since_signup_hours'] = (
        df['purchase_time'] - df['signup_time']
    ).dt.total_seconds() / 3600
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    return df