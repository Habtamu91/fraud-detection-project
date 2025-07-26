# Streamlit Deployment code 
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
from src.utils import convert_ip_to_integer, find_country_for_ip
import os

# === Load Model and Preprocessor ===
MODEL_PATH = "outputs/models/best_fraud_model_model.pkl"
PREPROCESSOR_PATH = "outputs/models/preprocessors/best_fraud_model_preprocessor.pkl"
IP_MAPPING_PATH = "data/raw/IpAddress_to_Country.csv"

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
ip_mapping_df = pd.read_csv(IP_MAPPING_PATH)

# Convert IP ranges
ip_mapping_df['lower_int'] = ip_mapping_df['lower_bound_ip_address'].apply(convert_ip_to_integer)
ip_mapping_df['upper_int'] = ip_mapping_df['upper_bound_ip_address'].apply(convert_ip_to_integer)

# === Streamlit App UI ===
st.set_page_config(page_title="Fraud Detection System", layout="centered")
st.title("🔍 Fraud Detection System")
st.markdown("Enter transaction details below to predict if it's fraudulent.")

# === Form Inputs ===
st.header("🧾 Transaction Details")
purchase_value = st.number_input("Purchase Value ($)", min_value=0.00, format="%.2f")
age = st.slider("User Age", min_value=18, max_value=90)

signup_date = st.date_input("Signup Date", value=datetime.today())
signup_hour = st.slider("Signup Hour", 0, 23)

purchase_date = st.date_input("Purchase Date", value=datetime.today())
purchase_hour = st.slider("Purchase Hour", 0, 23)

browser = st.selectbox("Browser", ["Chrome", "Safari", "IE", "Firefox"])
sex = st.radio("User Gender", ["M", "F"])
source = st.selectbox("Traffic Source", ["Ads", "SEO", "Direct"])
day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

ip_address = st.text_input("IP Address (e.g., 192.168.1.1)", value="192.168.1.1")

# === Prediction Logic ===
if st.button("Predict Fraud"):
    try:
        # Convert datetime inputs to datetime object
        signup_dt = pd.to_datetime(signup_date) + timedelta(hours=signup_hour)
        purchase_dt = pd.to_datetime(purchase_date) + timedelta(hours=purchase_hour)
        time_since_signup = (purchase_dt - signup_dt).total_seconds() / 3600

        # Get hour of purchase (redundant, but explicit)
        hour_of_day = purchase_dt.hour

        # Convert IP to country
        ip_int = convert_ip_to_integer(ip_address)
        country = find_country_for_ip(ip_int, ip_mapping_df)

        # Build feature dictionary
        input_data = {
            'purchase_value': purchase_value,
            'age': age,
            'time_since_signup': time_since_signup,
            'hour_of_day': hour_of_day,
            'browser': browser,
            'sex': sex,
            'source': source,
            'day_of_week': day_of_week,
            'country': country
        }

        # Create input DataFrame
        input_df = pd.DataFrame([input_data])

        # Transform with the preprocessor
        input_transformed = preprocessor.transform(input_df)

        # Make prediction
        pred_class = model.predict(input_transformed)[0]
        pred_proba = model.predict_proba(input_transformed)[0][1]

        # Display result
        if pred_class == 1:
            st.error(f"⚠️ This transaction is predicted to be **FRAUDULENT** with {pred_proba:.2%} confidence.")
        else:
            st.success(f"✅ This transaction is predicted to be **LEGITIMATE** with {(1 - pred_proba):.2%} confidence.")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
