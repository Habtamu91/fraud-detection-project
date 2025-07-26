---

# 🔐 Fraud Detection for E-Commerce and Bank Transactions.

This repository contains an end-to-end fraud detection project developed as part of the **10 Academy Week 8 & 9 Artificial Intelligence Mastery Challenge**.
The goal is to build accurate and explainable machine learning models that detect fraudulent activities in both **e-commerce** and **bank credit card** transactions.

---

## 🎯 Project Objectives.

* ✅ Load and explore real-world fraud datasets: **Fraud\_Data.csv** and **creditcard.csv**
* ✅ Handle missing values, correct data types, and merge datasets for geolocation analysis
* ✅ Engineer powerful features such as:

  * `time_since_signup`, `hour_of_day`, `day_of_week`
  * IP-to-country mapping
* ✅ Address **class imbalance** using SMOTE oversampling
* ✅ Train and compare two models:

  * Logistic Regression (baseline)
  * XGBoost Classifier (final)
* ✅ Evaluate models using `F1 Score`, `AUC-PR`, and Confusion Matrix
* ✅ Use **SHAP** for global and local interpretability
* ✅ Deploy the best model with saved weights for production

---

## 📁 Project Structure

```
fraud_detection_project/
├── data/
│   ├── raw/                  <- Original datasets (Fraud_Data, CreditCard)
│   └── processed/            <- Cleaned and feature-engineered data
├── notebooks/
│   ├── 01_eda_data_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_explainability.ipynb
├── outputs/
│   ├── models/               <- Trained model artifacts
│   ├── shap/                 <- SHAP plots and visualizations
│   └── logs/
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── shap_analysis.py
│   ├── train_model.py
│   ├── utils.py
│   └── __init__.py
├── tests/
├── visualizations/           <- Exploratory and SHAP visualizations
├── app/                      <- Deployment code (optional Flask or Streamlit)
├── environment.yml
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Setup Instructions

Create the conda environment and activate it:

```bash
conda env create -f environment.yml
conda activate fraud-detection-env
```

Install required packages (optional):

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Step 1: Data Preprocessing

```python
from src.preprocessing import preprocess_fraud_data
X_train, X_test, y_train, y_test = preprocess_fraud_data("data/raw/Fraud_Data.csv")
```

### Step 2: Model Training

```python
from src.train_model import train_and_evaluate_models
train_and_evaluate_models("data/processed/fraud_preprocessed.pkl", model_name="fraud")
```

### Step 3: SHAP Explainability

```python
from src.shap_analysis import explain_with_shap
explain_with_shap("outputs/models/best_fraud_model.pkl", 
                  "data/processed/fraud_preprocessed.pkl", 
                  "outputs/shap/fraud_shap_summary.png")
```

---

## 📊 Model Performance

| Dataset     | Model               | F1 Score | AUC-PR | Notes                   |
| ----------- | ------------------- | -------- | ------ | ----------------------- |
| E-Commerce  | Logistic Regression | 0.27     | 0.67   | Baseline                |
| E-Commerce  | XGBoost Classifier  | 0.68     | 0.76   | ✅ Selected Model        |
| Credit Card | Logistic Regression | 0.83     | 0.84   | Competitive but simpler |
| Credit Card | XGBoost Classifier  | 0.92     | 0.95   | ✅ Best overall model    |

✔️ XGBoost was selected for both datasets due to superior performance across key metrics.

---

## 📍 SHAP Explainability

SHAP was used to answer two key questions:

* 🔍 **Which features are most influential in predicting fraud?**
* 🧠 **How do model decisions change based on individual transactions?**

Results:

* `time_since_signup`, `browser`, and `source` were influential in e-commerce
* `V4`, `V14`, and `Amount` were important drivers in credit card fraud detection

SHAP visualizations are saved in: `outputs/shap/`

---

## 🧾 Deployment

The best models have been serialized and saved under `outputs/models/` for production use.

For future work, these models can be deployed via:

* Flask or FastAPI (for API endpoints)
* Streamlit (for UI interfaces)

---

## 🔗 GitHub Repository

[📂 View on GitHub]
(https://github.com/Habtamu91/fraud-detection-project)

---

📁 Note:
Large files like `creditcard.csv` and model artifacts are excluded due to GitHub's 100MB limit.
🔗 Download from: [https://mail.google.com/mail/u/0/#search/Technical+/FMfcgzQbgJPZnSzJSXbcqGLVjVrshKTg]

## 👤 Author

**Habtamu Belay Tessema**
Data Science Student, Bahir Dar University
📫 Email: [habtamubelay543@gmail.com]

---

## 🙏 Acknowledgements

* 💼 Organized by **10 Academy**
* 👩‍🏫 Mentors: Mahlet, Rediet, Rehmet, Kerod
* 📚 Data Source: [Kaggle Datasets](https://www.kaggle.com/datasets)

---