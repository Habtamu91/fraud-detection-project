import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

def load_preprocessed_data(pkl_path):
    return joblib.load(pkl_path)

def explain_with_shap(model_path, data_path, output_image_path):
    model = joblib.load(model_path)
    X_train, X_test, _, _ = load_preprocessed_data(data_path)

    # Use SHAP TreeExplainer (works best with tree-based models like XGBoost)
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)

    # Save summary plot
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(output_image_path, bbox_inches='tight')
    plt.close()
    print(f"✅ SHAP summary plot saved to: {output_image_path}")
