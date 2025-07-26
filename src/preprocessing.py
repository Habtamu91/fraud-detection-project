# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path

# Define these at the module level or within the function where they're used
NUMERICAL_COLS = ['purchase_value', 'age', 'time_since_signup', 'hour_of_day']
CATEGORICAL_COLS = ['browser', 'sex', 'source', 'day_of_week']

def convert_ip_to_integer(ip_str):
    """Convert IP string to integer (needed for IP mapping)"""
    try:
        parts = list(map(int, ip_str.split('.')))
        return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
    except Exception:
        return np.nan

def load_and_merge_data(fraud_data_path, ip_mapping_path=None):
    """Load and merge data with IP mapping if available"""
    df = pd.read_csv(fraud_data_path)
    
    if ip_mapping_path:
        ip_df = pd.read_csv(ip_mapping_path)
        df = map_ip_to_country(df, ip_df)
    
    return df

def map_ip_to_country(fraud_df, ip_df):
    """Map IP addresses to countries"""
    fraud_df['ip_int'] = fraud_df['ip_address'].apply(convert_ip_to_integer)
    ip_df['lower_int'] = ip_df['lower_bound_ip_address'].apply(convert_ip_to_integer)
    ip_df['upper_int'] = ip_df['upper_bound_ip_address'].apply(convert_ip_to_integer)
    
    # Sort for efficient mapping
    ip_df = ip_df.sort_values('lower_int')
    
    # Map IP to country
    fraud_df['country'] = fraud_df['ip_int'].apply(
        lambda x: ip_df.loc[
            (ip_df['lower_int'] <= x) & (ip_df['upper_int'] >= x), 'country'
        ].values[0] if not ip_df[
            (ip_df['lower_int'] <= x) & (ip_df['upper_int'] >= x)
        ].empty else 'Unknown'
    )
    
    return fraud_df.drop(columns=['ip_address', 'ip_int'])

def engineer_features(df):
    """Create derived features"""
    # Convert timestamps
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    # Time-based features
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.day_name()
    
    # Drop original columns
    df = df.drop(columns=['signup_time', 'purchase_time', 'device_id', 'user_id'], errors='ignore')
    
    return df

def get_preprocessor(include_country=True):
    """Return configured preprocessor"""
    categorical_cols = CATEGORICAL_COLS.copy()
    if include_country:
        categorical_cols.append('country')
    
    return ColumnTransformer([
        ('num', StandardScaler(), NUMERICAL_COLS),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

def preprocess_fraud_data(fraud_data_path, ip_mapping_path=None, save_artifacts=False):
    """Full preprocessing pipeline"""
    # Load and merge data
    df = load_and_merge_data(fraud_data_path, ip_mapping_path)
    
    # Feature engineering
    df = engineer_features(df)
    
    # Split features and target
    target = 'class'
    X = df.drop(columns=[target])
    y = df[target]
    
    # Initialize preprocessor
    preprocessor = get_preprocessor(include_country=ip_mapping_path is not None)
    
    # Process features
    X_processed = preprocessor.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Save artifacts if requested
    if save_artifacts:
        # Create outputs directory structure
        output_dir = Path("outputs/models/preprocessors")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save preprocessor
        joblib.dump(preprocessor, output_dir/"fraud_preprocessor.pkl")
        
        # Generate and save feature names
        feature_names = NUMERICAL_COLS.copy()
        if hasattr(preprocessor.named_transformers_['cat'], 'get_feature_names_out'):
            feature_names.extend(
                preprocessor.named_transformers_['cat'].get_feature_names_out(
                    CATEGORICAL_COLS + (['country'] if ip_mapping_path else [])
                )
            )
        pd.Series(feature_names).to_csv(output_dir/"feature_names.csv", index=False)
        
        print(f"✅ Preprocessing artifacts saved to: {output_dir}")
    
    return X_train_balanced, X_test, y_train_balanced, y_test, preprocessor