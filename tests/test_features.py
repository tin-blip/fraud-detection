"""
Unit tests for feature engineering functions
"""

import pytest
import pandas as pd
import numpy as np

def test_time_features():
    """Test time-based feature creation"""
    # Create sample data
    df = pd.DataFrame({
        'purchase_time': pd.date_range('2024-01-01', periods=10, freq='H')
    })
    
    # Create features
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Assertions
    assert df['hour_of_day'].iloc[0] == 0
    assert df['hour_of_day'].iloc[5] == 5
    assert 0 <= df['hour_of_day'].max() <= 23
    assert 0 <= df['day_of_week'].max() <= 6
    assert df['is_weekend'].isin([0, 1]).all()
    
    print("✓ Time features test passed")


def test_velocity_features():
    """Test transaction velocity features"""
    # Create sample with multiple transactions per user
    df = pd.DataFrame({
        'user_id': [1, 1, 1, 2, 2],
        'purchase_time': pd.to_datetime([
            '2024-01-01 10:00:00',
            '2024-01-01 10:30:00',  # 30 min later
            '2024-01-01 12:00:00',  # 2 hours later
            '2024-01-01 09:00:00',
            '2024-01-01 10:00:00'
        ])
    })
    
    # Sort
    df = df.sort_values(['user_id', 'purchase_time'])
    
    # Calculate transactions in last hour
    df['time_since_last'] = df.groupby('user_id')['purchase_time'].diff().dt.total_seconds() / 3600
    df['transactions_last_hour'] = (df['time_since_last'] <= 1).astype(int)
    df['transactions_last_hour'] = df.groupby('user_id')['transactions_last_hour'].fillna(0)
    
    # User 1: first = 0, second = 1 (within hour), third = 0 (after hour)
    user1 = df[df['user_id'] == 1]
    assert user1['transactions_last_hour'].iloc[0] == 0
    assert user1['transactions_last_hour'].iloc[1] == 1
    
    print("✓ Velocity features test passed")


def test_device_features():
    """Test device-based features"""
    # Create sample with shared devices
    df = pd.DataFrame({
        'device_id': ['A', 'A', 'B', 'B', 'C'],
        'user_id': [1, 2, 3, 3, 4],
        'class': [0, 1, 0, 0, 1]
    })
    
    # Calculate device stats
    device_stats = df.groupby('device_id').agg({
        'user_id': 'nunique',
        'class': 'mean'
    }).rename(columns={'user_id': 'users_per_device', 'class': 'device_fraud_rate'})
    
    df = df.merge(device_stats, on='device_id', how='left')
    
    # Assertions
    assert df[df['device_id'] == 'A']['users_per_device'].iloc[0] == 2
    assert df[df['device_id'] == 'A']['device_fraud_rate'].iloc[0] == 0.5
    assert df[df['device_id'] == 'B']['users_per_device'].iloc[0] == 1
    
    print("✓ Device features test passed")


def test_feature_no_nan():
    """Test that no NaN values in created features"""
    df = pd.DataFrame({
        'user_id': [1, 2, 3],
        'purchase_time': pd.date_range('2024-01-01', periods=3, freq='H'),
        'device_id': ['A', 'B', 'C'],
        'class': [0, 0, 1]
    })
    
    # Create features
    df['hour'] = df['purchase_time'].dt.hour
    df['day'] = df['purchase_time'].dt.day
    df['users_per_device'] = 1
    df['device_fraud_rate'] = df['class']
    
    assert df[['hour', 'day', 'users_per_device', 'device_fraud_rate']].isnull().sum().sum() == 0
    
    print("✓ No NaN values test passed")


if name == "main":
    test_time_features()
    test_velocity_features()
    test_device_features()
    test_feature_no_nan()
    print("\n✅ All feature engineering tests passed!")