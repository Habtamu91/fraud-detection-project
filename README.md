---

###  `README.md` (Updated for Interim 2)

# 🔐 Fraud Detection for E-Commerce and Bank Transactions

This repository contains an end-to-end fraud detection project developed as part of the **10 Academy Week 8 & 9 Artificial Intelligence Mastery Challenge**.  
The goal is to detect fraudulent activities in both e-commerce and bank transaction datasets using robust machine learning models and model explainability tools.

---

## 🎯 Project Objectives

- ✅ Load and explore two real-world fraud datasets
- ✅ Clean and preprocess time and categorical data
- ✅ Engineer informative features like `time_since_signup`, `hour_of_day`, and `day_of_week`
- ✅ Address class imbalance using **SMOTE** oversampling
- ✅ Train and compare models:
  - Logistic Regression (Baseline)
  - XGBoost Classifier (Advanced)
- ✅ Evaluate using `F1 Score`, `AUC-PR`, and Confusion Matrix
- ✅ Use **SHAP** for model interpretability
- ✅ Save the best-performing model for deployment

---

## 📂 Folder Structure

```

fraud_detection_project/
├── data/
│   ├── raw/              <- Original CSV files
│   └── processed/        <- Cleaned and transformed data
├── notebooks/            <- Jupyter notebooks for each pipeline stage
│   ├── 01_eda_data_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_explainability.ipynb
├── outputs/
│   ├── models/           <- Saved best models
│   ├── shap/             <- SHAP visualizations
│   └── logs/
├── src/                  <- Python source code modules
│   ├── config.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── shap_analysis.py
│   ├── train_model.py
│   ├── utils.py
│   └── __init__.py
├── tests/
├── visualizations/
├── README.md
├── environment.yml
├── requirements.txt
└── .gitignore

````

---

## How to Run the Project

Make sure you have the environment set up from `environment.yml`.

```bash
conda env create -f environment.yml
conda activate fraud-detection-env
````

### Step 1: Preprocessing (in notebook or script)

```python
# From notebooks/02_feature_engineering.ipynb
from src.preprocessing import preprocess_fraud_data
X_train, X_test, y_train, y_test = preprocess_fraud_data("data/raw/Fraud_Data.csv")
```

### Step 2: Model Training

```python
# From notebooks/03_model_training.ipynb
from src.train_model import train_and_evaluate_models
train_and_evaluate_models("data/processed/fraud_preprocessed.pkl", model_name="fraud")
```

### Step 3: SHAP Explainability

```python
# From notebooks/04_model_explainability.ipynb
from src.shap_analysis import explain_with_shap
explain_with_shap("outputs/models/best_fraud_model.pkl", 
                  "data/processed/fraud_preprocessed.pkl", 
                  "outputs/shap/fraud_shap_summary.png")
```

---

## 📈 Model Results (E-Commerce Fraud Dataset)

| Model               | F1 Score | AUC-PR | Notes                    |
| ------------------- | -------- | ------ | ------------------------ |
| Logistic Regression | 0.27     | 0.67   | Baseline, low precision  |
| XGBoost Classifier  | 0.68     | 0.76   | ✅ Best performer (saved) |

> The XGBoost model is selected as the final model due to better balance of recall and precision.

---

## 🔍 SHAP Explainability

SHAP summary plots were generated to identify:

* 💡 Which features most influence fraud predictions
* 🧠 How model decisions vary for individual transactions

SHAP plots are saved under `outputs/shap/`.

---

## 📎 GitHub Repository

> 🛠 [https://github.com/Habtamu91/fraud-detection-project](https://github.com/Habtamu91/fraud-detection-project)

---

## 👨‍💻 Author

**Habtamu Belay Tessema**
 Data Science Student
Bahir Dar University

---

## 🙏 Acknowledgements

* 💼 Project organized by **10 Academy**
* 👩‍🏫 Mentors: Mahlet, Rediet, Rehmet, Kerod

````


