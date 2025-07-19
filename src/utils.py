# Utility functions like logging, metrics
import os
import joblib
import logging
import pandas as pd
import numpy as np


# 1. Set up logging
def get_logger(name=__name__):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )
    return logging.getLogger(name)


# 2. File saving/loading
def save_pickle(obj, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(obj, filepath)
    print(f"✅ Saved to {filepath}")

def load_pickle(filepath):
    if os.path.exists(filepath):
        return joblib.load(filepath)
    else:
        raise FileNotFoundError(f"❌ File not found: {filepath}")


# 3. Ensure directory exists
def ensure_dir_exists(path):
    os.makedirs(path, exist_ok=True)


# 4. IP Address conversion
def convert_ip_to_integer(ip_str):
    """Convert IP address string to integer"""
    try:
        parts = list(map(int, ip_str.split('.')))
        return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
    except Exception:
        return np.nan


# 5. Country lookup using IP
def find_country_for_ip(ip_int, ip_df):
    """Return country name if IP is in a known range"""
    match = ip_df[(ip_df['lower_int'] <= ip_int) & (ip_df['upper_int'] >= ip_int)]
    if not match.empty:
        return match.iloc[0]['country']
    return 'Unknown'
