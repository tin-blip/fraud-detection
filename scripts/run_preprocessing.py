"""
Run the complete preprocessing pipeline from command line
Usage: python scripts/run_preprocessing.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(file))))

import pandas as pd
import numpy as np
import ipaddress
from src.preprocess import load_and_clean_data, create_time_features
from src.features import create_velocity_features, create_device_features

def ip_to_int(ip_string):
    """Convert IP address to integer"""
    try:
        return int(ipaddress.IPv4Address(str(ip_string)))
    except:
        return None

def main():
    print("=" * 60)
    print("STARTING PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n1. Loading raw data...")
    df = pd.read_csv('data/raw/Fraud_Data.csv')
    print(f"   Loaded {len(df)} transactions")
    
    # Step 2: Convert dates
    print("\n2. Converting date columns...")
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    # Step 3: IP to country mapping
    print("\n3. Mapping IPs to countries...")
    ip_mapping = pd.read_csv('data/raw/IpAddress_to_Country.csv')
    
    # Convert IPs
    df['ip_int'] = df['ip_address'].astype('int64')
    ip_mapping['lower_int'] = ip_mapping['lower_bound_ip_address'].apply(ip_to_int)
    ip_mapping['upper_int'] = ip_mapping['upper_bound_ip_address'].apply(ip_to_int)
    ip_mapping = ip_mapping.dropna()
    
    # Merge
    df_sorted = df.sort_values('ip_int')
    ip_sorted = ip_mapping.sort_values('lower_int')
    df_mapped = pd.merge_asof(df_sorted, ip_sorted, left_on='ip_int', right_on='lower_int', direction='backward')
    df_mapped = df_mapped[(df_mapped['ip_int'] >= df_mapped['lower_int']) & (df_mapped['ip_int'] <= df_mapped['upper_int'])]
    print(f"   Mapped {len(df_mapped)} transactions to countries")
    
    # Step 4: Feature engineering
    print("\n4. Engineering features...")
    df_mapped['time_since_signup_hours'] = (df_mapped['purchase_time'] - df_mapped['signup_time']).dt.total_seconds() / 3600
    df_mapped['hour_of_day'] = df_mapped['purchase_time'].dt.hour
    df_mapped['day_of_week'] = df_mapped['purchase_time'].dt.dayofweek
    
    # Step 5: Save processed data
    print("\n5. Saving processed data...")
    df_mapped.to_csv('data/processed/fraud_data_processed.csv', index=False)
    print(f"   Saved to data/processed/fraud_data_processed.csv")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    
    return df_mapped

if name == "main":
    df = main()
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Fraud rate: {df['class'].mean():.4f}")