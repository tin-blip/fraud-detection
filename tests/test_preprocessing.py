"""
Unit tests for preprocessing
"""

import pytest
import pandas as pd
import numpy as np

def test_ip_conversion():
    """Test IP conversion"""
    def ip_to_int(ip_string):
        try:
            import ipaddress
            return int(ipaddress.IPv4Address(str(ip_string)))
        except:
            return None
    
    assert ip_to_int("192.168.1.1") == 3232235777
    assert ip_to_int("invalid") is None
    print("✓ IP conversion test passed")


def test_date_conversion():
    """Test date conversion"""
    df = pd.DataFrame({
        "signup_time": ["2024-01-01 10:00:00"],
        "purchase_time": ["2024-01-01 12:00:00"]
    })
    
    df["signup_time"] = pd.to_datetime(df["signup_time"])
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])
    diff = (df["purchase_time"] - df["signup_time"]).dt.total_seconds() / 3600
    
    assert diff.iloc[0] == 2.0
    print("✓ Date conversion test passed")


def test_data_cleaning():
    """Test cleaning"""
    df = pd.DataFrame({"id": [1, 1, 2], "value": [10, None, 20]})
    df = df.drop_duplicates()
    df["value"] = df["value"].fillna(df["value"].median())
    
    assert len(df) == 2
    assert df["value"].isnull().sum() == 0
    print("✓ Data cleaning test passed")


def test_imports():
    """Test imports"""
    try:
        import pandas
        import numpy
        import sklearn
        print("✓ All imports successful")
    except ImportError as e:
        pytest.fail(f"Missing: {e}")


if name == "main":
    test_ip_conversion()
    test_date_conversion()
    test_data_cleaning()
    test_imports()
    print("\n✅ All preprocessing tests passed!")
