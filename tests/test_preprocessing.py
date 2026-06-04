"""
Unit tests for preprocessing functions
"""

import pytest
import pandas as pd
import numpy as np

def test_ip_conversion():
    """Test IP address conversion"""
    def ip_to_int(ip_string):
        try:
            import ipaddress
            return int(ipaddress.IPv4Address(str(ip_string)))
        except:
            return None
    
    # Test valid IP
    assert ip_to_int('192.168.1.1') == 3232235777
    
    # Test invalid IP
    assert ip_to_int('invalid') is None
    
    # Test None input
    assert ip_to_int(None) is None
    
    print("✓ IP conversion test passed")


def test_date_conversion():
    """Test date conversion"""
    # Create sample data
    df = pd.DataFrame({
        'signup_time': ['2024-01-01 10:00:00', '2024-01-02 11:00:00'],
        'purchase_time': ['2024-01-01 12:00:00', '2024-01-03 13:00:00']
    })
    
    # Convert to datetime
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    # Calculate time difference
    df['time_diff_hours'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
    
    # Assertions
    assert df['signup_time'].dtype == 'datetime64[ns]'
    assert df['purchase_time'].dtype == 'datetime64[ns]'
    assert df['time_diff_hours'].iloc[0] == 2.0  # 2 hours difference
    
    print("✓ Date conversion test passed")


def test_data_cleaning():
    """Test basic data cleaning"""
    # Create sample with duplicates and nulls
    df = pd.DataFrame({
        'id': [1, 1, 2, 3],
        'value': [10, 10, None, 20],
        'class': [0, 0, 1, 1]
    })
    
    # Remove duplicates
    df = df.drop_duplicates()
    assert len(df) == 3
    
    # Fill nulls
    df['value'] = df['value'].fillna(df['value'].median())
    assert df['value'].isnull().sum() == 0
    
    print("✓ Data cleaning test passed")


def test_imports():
    """Test that required packages are installed"""
    try:
        import pandas
        import numpy
        import sklearn
        import imblearn
        print("✓ All core packages available")
    except ImportError as e:
        pytest.fail(f"Missing package: {e}")


if name == "main":
    test_ip_conversion()
    test_date_conversion()
    test_data_cleaning()
    test_imports()
    print("\n✅ All preprocessing tests passed!")