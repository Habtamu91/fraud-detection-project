import pandas as pd
import numpy as np
from datetime import datetime

def engineer_features_fraud_data(df):
    # Ensure timestamps are datetime objects
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])

    # Calculate time difference in hours between signup and purchase
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600

    # Extract hour and day of week
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.day_name()

    # Drop raw datetime columns if no longer needed
    df.drop(['signup_time', 'purchase_time'], axis=1, inplace=True)

    return df

def convert_ip_to_integer(ip_str):
    """Convert IP string to integer (needed for IP mapping)"""
    try:
        parts = list(map(int, ip_str.split('.')))
        return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
    except Exception:
        return np.nan

def map_ip_to_country(df, ip_mapping_df):
    """
    Merges fraud dataframe with IP-to-country mapping.
    Both IPs must be converted to integers before joining.
    """
    df['ip_int'] = df['ip_address'].apply(convert_ip_to_integer)
    ip_mapping_df['lower_int'] = ip_mapping_df['lower_bound_ip_address'].apply(convert_ip_to_integer)
    ip_mapping_df['upper_int'] = ip_mapping_df['upper_bound_ip_address'].apply(convert_ip_to_integer)

    # Sort for efficient mapping
    ip_mapping_df.sort_values(by='lower_int', inplace=True)

    # Map IP to country using interval join logic
    df['country'] = df['ip_int'].apply(lambda x: find_country_for_ip(x, ip_mapping_df))
    df.drop(columns=['ip_address', 'ip_int'], inplace=True)
    return df

def find_country_for_ip(ip, ip_df):
    """Helper to find which country a given IP belongs to"""
    match = ip_df[(ip_df['lower_int'] <= ip) & (ip_df['upper_int'] >= ip)]
    if not match.empty:
        return match.iloc[0]['country']
    return 'Unknown'
if __name__ == "__main__":
    import os

    data_path = "data/raw/Fraud_Data.csv"  # adjust if your file is elsewhere
    if os.path.exists(data_path):
        print("✅ Loading data...")
        df = pd.read_csv(data_path)
        print("📦 Original columns:", df.columns.tolist())

        df = engineer_features_fraud_data(df)
        print("\n🧠 Feature engineering complete.")
        print("📊 New columns:", df.columns.tolist())
        print(df.head())
    else:
        print(f"❌ File not found at {data_path}")
