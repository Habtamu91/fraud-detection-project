# Configuration variables like paths, params
import os

# Base project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raw data paths
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
FRAUD_DATA_PATH = os.path.join(RAW_DATA_DIR, "Fraud_Data.csv")
CREDITCARD_DATA_PATH = os.path.join(RAW_DATA_DIR, "creditcard.csv")
IP_MAPPING_PATH = os.path.join(RAW_DATA_DIR, "IpAddress_to_Country.csv")

# Processed data paths
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
PROCESSED_FRAUD_DATA = os.path.join(PROCESSED_DATA_DIR, "processed_fraud.pkl")
PROCESSED_CREDITCARD_DATA = os.path.join(PROCESSED_DATA_DIR, "processed_creditcard.pkl")

# Output directories
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "models")
SHAP_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "shap")

# Best model paths
BEST_FRAUD_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, "best_fraud_model.pkl")
BEST_CREDIT_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, "best_creditcard_model.pkl")

# SHAP plot paths
FRAUD_SHAP_PLOT = os.path.join(SHAP_OUTPUT_DIR, "fraud_shap_summary.png")
CREDIT_SHAP_PLOT = os.path.join(SHAP_OUTPUT_DIR, "creditcard_shap_summary.png")
