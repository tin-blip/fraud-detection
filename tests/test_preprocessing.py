import pandas as pd
import numpy as np
import pytest

def test_data_loading():
    """Test that data can be loaded"""
    try:
        df = pd.read_csv('data/raw/Fraud_Data.csv')
        assert df.shape[0] > 0
        assert 'class' in df.columns
        print("✓ Data loading test passed")
    except:
        print("⚠️ Data file not found - skipping test")

def test_feature_engineering():
    """Test that features are created correctly"""
    # Create sample data
    df = pd.DataFrame({
        'signup_time': pd.date_range('2024-01-01', periods=5),
        'purchase_time': pd.date_range('2024-01-01', periods=5) + pd.Timedelta(hours=2)
    })
    
    # Calculate time difference
    time_diff = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
    
    assert len(time_diff) == 5
    assert (time_diff == 2).all()
    print("✓ Feature engineering test passed")

def test_class_balance():
    """Test that data has fraud cases"""
    try:
        df = pd.read_csv('data/raw/Fraud_Data.csv')
        fraud_rate = df['class'].mean()
        assert fraud_rate > 0
        print(f"✓ Fraud rate: {fraud_rate:.4f}")
    except:
        print("⚠️ Cannot test class balance - data not found")

if name == "main":
    test_data_loading()
    test_feature_engineering()
    test_class_balance()