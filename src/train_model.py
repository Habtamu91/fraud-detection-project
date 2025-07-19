import joblib
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_auc_score

def load_data(filepath):
    return joblib.load(filepath)

def train_and_evaluate_models(data_path, model_name="fraud"):
    X_train, X_test, y_train, y_test = load_data(data_path)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)

    print("\n📊 Logistic Regression Report")
    print(classification_report(y_test, y_pred_lr))
    print("F1:", f1_score(y_test, y_pred_lr))
    print("AUC-PR:", roc_auc_score(y_test, y_pred_lr))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

    print("\n📊 XGBoost Report")
    print(classification_report(y_test, y_pred_xgb))
    print("F1:", f1_score(y_test, y_pred_xgb))
    print("AUC-PR:", roc_auc_score(y_test, y_pred_xgb))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

    best_model = xgb if f1_score(y_test, y_pred_xgb) > f1_score(y_test, y_pred_lr) else lr
    model_path = f"../outputs/models/best_{model_name}_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"✅ Best model saved to: {model_path}")
