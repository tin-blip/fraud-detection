"""Feature engineering utilities"""

import pandas as pd

def create_velocity_features(df):
    """Create transaction velocity features"""
    df = df.sort_values(['user_id', 'purchase_time'])
    
    df['transactions_last_hour'] = df.groupby('user_id')['purchase_time'].transform(
        lambda x: x.diff().dt.total_seconds().fillna(999999) < 3600
    ).astype(int)
    
    return df

def create_device_features(df):
    """Create device-based features"""
    device_stats = df.groupby('device_id').agg({
        'user_id': 'nunique',
        'class': 'mean'
    }).rename(columns={'user_id': 'users_per_device', 'class': 'device_fraud_rate'})
    
    return df.merge(device_stats, on='device_id', how='left')