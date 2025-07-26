# src/train_model.py
import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, f1_score, 
                           confusion_matrix, roc_auc_score, 
                           average_precision_score)
from src.preprocessing import preprocess_fraud_data

def save_model_artifacts(model, preprocessor, model_name):
    """Save model and preprocessing artifacts"""
    # Create output directories
    models_dir = Path("outputs/models")
    preprocessors_dir = Path("outputs/models/preprocessors")
    models_dir.mkdir(parents=True, exist_ok=True)
    preprocessors_dir.mkdir(parents=True, exist_ok=True)
    
    # Define paths
    model_path = models_dir/f"{model_name}_model.pkl"
    preprocessor_path = preprocessors_dir/f"{model_name}_preprocessor.pkl"
    
    # Save artifacts
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Preprocessor saved to: {preprocessor_path}")
    
    return model_path, preprocessor_path

def evaluate_model(model, X_test, y_test, model_name):
    """Generate evaluation metrics"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n📊 {model_name} Evaluation Report")
    print("="*50)
    print(classification_report(y_test, y_pred))
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"PR AUC: {average_precision_score(y_test, y_proba):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return {
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'pr_auc': average_precision_score(y_test, y_proba)
    }

def train_and_evaluate_models(fraud_data_path, ip_mapping_path=None):
    """Main training pipeline"""
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_fraud_data(
        fraud_data_path=fraud_data_path,
        ip_mapping_path=ip_mapping_path,
        save_artifacts=True
    )
    
    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, 
            random_state=42,
            class_weight='balanced'
        ),
        "XGBoost": XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])
        )
    }
    
    # Train and evaluate
    results = {}
    for name, model in models.items():
        print(f"\n🚀 Training {name}...")
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_test, y_test, name)
    
    # Select and save best model
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_model = models[best_model_name]
    
    model_path, preprocessor_path = save_model_artifacts(
        best_model, 
        preprocessor,
        f"best_fraud_model"
    )
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"F1 Score: {results[best_model_name]['f1']:.4f}")
    print(f"ROC AUC: {results[best_model_name]['roc_auc']:.4f}")
    print(f"PR AUC: {results[best_model_name]['pr_auc']:.4f}")
    
    return best_model, preprocessor, results

if __name__ == "__main__":
    # Example usage
    print("Training fraud detection model...")
    train_and_evaluate_models(
        fraud_data_path="data/processed/fraud_cleaned.csv",
        ip_mapping_path="data/raw/IpAddress_to_Country.csv"
    )