# src/predict.py
import pandas as pd
import joblib
import numpy as np
from .preprocessing import convert_ip_to_integer, engineer_features

class FraudPredictor:
    def __init__(self, model_path, preprocessor_path):
        """Initialize with trained model and preprocessor"""
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        
    def prepare_input(self, input_dict):
        """Convert raw input to properly formatted DataFrame"""
        # Create DataFrame from input
        input_df = pd.DataFrame([input_dict])
        
        # Apply same feature engineering as training
        if 'ip_address' in input_df.columns:
            input_df['ip_int'] = input_df['ip_address'].apply(convert_ip_to_integer)
        
        # Ensure all expected columns are present
        expected_cols = [
            'purchase_value', 'age', 'time_since_signup', 'hour_of_day',
            'browser', 'sex', 'source', 'day_of_week', 'ip_int'
        ]
        
        for col in expected_cols:
            if col not in input_df.columns:
                input_df[col] = np.nan  # Fill missing with NaN
                
        return input_df
    
    def predict(self, input_data):
        """Make prediction from raw input"""
        try:
            # Prepare features
            features = self.prepare_input(input_data)
            
            # Apply preprocessing
            processed_features = self.preprocessor.transform(features)
            
            # Make prediction
            prediction = self.model.predict(processed_features)[0]
            probability = self.model.predict_proba(processed_features)[0][1]
            
            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'status': 'success'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'features': features.to_dict() if 'features' in locals() else None
            }