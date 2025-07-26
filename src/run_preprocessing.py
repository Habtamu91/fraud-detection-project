# src/run_preprocessing.py
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import preprocess_fraud_data
import joblib

def main():

    # Run preprocessing
    X_train, X_test, y_train, y_test, preprocessor = preprocess_fraud_data(
        fraud_data_path="data/raw/Fraud_Data.csv",
        ip_mapping_path="data/raw/IpAddress_to_Country.csv", 
        save_artifacts=True
    )
    
    # Save processed data
    joblib.dump((X_train, y_train), "data/processed/train_data.pkl")
    joblib.dump((X_test, y_test), "data/processed/test_data.pkl")
    print("✅ Preprocessing complete. Artifacts saved.")

if __name__ == "__main__":
    main()
    