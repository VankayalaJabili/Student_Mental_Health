import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# 1. Load all your artifacts
model = joblib.load('../models/master_xgb.pkl')
scaler = joblib.load('../models/scaler.pkl')
explainer = joblib.load('../models/shap_explainer.pkl')

print("Artifacts loaded successfully.")

# 2. Load your raw CSV data
# Make sure to put the correct path to your 25k dataset here!
data = pd.read_csv('DF_Scores_Imp.csv') 

# Dynamically sample up to 500 rows for clean visuals
n_samples = min(500, len(data))
data = data.sample(n=n_samples, random_state=42)

# Clean out the UID and index
if 'Unnamed: 0' in data.columns:
    X_raw = data.drop(columns=['Unnamed: 0', 'UID'])
else:
    X_raw = data.drop(columns=['UID'])

# 3. CRITICAL FIX: Make it 10 features!
# Your model expects 10 columns. We will append 5 "delta" columns filled with 0s 
# so the model shape matches perfectly (5 raw + 5 deltas = 10).
base_cols = X_raw.columns.tolist()
for col in base_cols:
    X_raw[f"{col}_delta"] = 0.0

feature_names = X_raw.columns.tolist()
print(f"Final 10 Features: {feature_names}")

# 4. Scale the data using your scaler!
X_scaled = scaler.transform(X_raw)

# 5. Generate SHAP values
print("Calculating SHAP values...")
shap_values = explainer.shap_values(X_scaled)

# ---------------------------------------------------------
# 6. Generate Global Summary Plot (For Section V)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
# We pass X_scaled so the math works, but feature_names so the labels look good in the paper
shap.summary_plot(shap_values, X_scaled, feature_names=feature_names, show=False)
plt.title("Global Feature Importance (SHAP Summary)", pad=20)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Global Plot: shap_summary.png")

# ---------------------------------------------------------
# 7. Generate Local Waterfall Plot (For Case Study)
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))

class_index = 3 # Let's explain class 3 (Severe Risk)
user_index = 0  # For the very first user in our 500 sample

exp = shap.Explanation(
    values=shap_values[class_index][user_index], 
    base_values=explainer.expected_value[class_index], 
    data=X_scaled[user_index], 
    feature_names=feature_names
)

shap.waterfall_plot(exp, show=False)
plt.tight_layout()
plt.savefig('shap_local.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Local Plot: shap_local.png")