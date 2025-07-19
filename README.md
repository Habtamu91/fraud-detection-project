# Fraud Detection for E-Commerce and Bank Transactions

This project was built as part of the **10 Academy Week 8 & 9 Challenge**.  
Its objective is to improve fraud detection in both e-commerce and bank transaction datasets using advanced machine learning techniques and explainability tools.


## Project Goals

- Clean and explore fraud transaction datasets
- Engineer time-based and frequency-based features
- Handle severe class imbalance with SMOTE
- Build and compare ML models (Logistic Regression, XGBoost)
- Use SHAP to explain predictions
- Save best model for deployment

    ## How to Run

- Run Preprocessing + Model Training from Notebook:

# In notebooks/02_feature_engineering.ipynb
from preprocessing import preprocess_fraud_data
X_train, X_test, y_train, y_test = preprocess_fraud_data("data/processed/fraud_cleaned.csv")

# In notebooks/03_model_training.ipynb
from train_model import train_and_evaluate_models
train_and_evaluate_models("data/processed/fraud_preprocessed.pkl")

# In notebooks/04_model_explainability.ipynb
from shap_analysis import explain_with_shap
explain_with_shap("outputs/models/best_fraud_model.pkl", "data/processed/fraud_preprocessed.pkl")

📈 Model Results (Fraud Dataset)

Model	F1 Score	AUC-PR	Notes
Logistic Regression	0.27	0.67	High recall, low precision
XGBoost	0.68	0.76	Best performer (saved model)

🔍 SHAP Explainability
We used SHAP to understand:

Which features drive fraud predictions

How individual samples are classified


## Credits
Built by :  Habtamu Belay Tessema ,  during 10 Academy - Week 8&9 AI Mastery Challenge.

Special thanks to:

- 10 Academy team




