import sys
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Add src to path so we can import the preprocessing module
sys.path.append(os.path.dirname(__file__))
from preprocessing import TelecomDataCleaner

# --- Configurations ---
DATA_PATH = 'data/raw/telecom_churn.csv'
MODEL_PATH = 'models/churn_model.pkl'
OUTPUT_DIR = 'visualizations'

# Set modern plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("notebook", font_scale=1.2)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_graphs():
    print("📊 Loading data and model...")
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    # Prepare data (same split as training)
    y = df['Churn'].astype(int)
    X = df.drop(columns=['Churn'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get Predictions
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= 0.30).astype(int) # Using your custom 30% threshold

    # ==========================================
    # GRAPH 1: Explainable AI Feature Importance
    # ==========================================
    print("📈 Generating Feature Importance Chart...")
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    coefficients = model.named_steps['classifier'].coef_[0]
    
    importance_df = pd.DataFrame({'Feature': feature_names, 'Impact': coefficients})
    importance_df['Absolute_Impact'] = importance_df['Impact'].abs()
    importance_df = importance_df.sort_values(by='Absolute_Impact', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    colors = ['#e74c3c' if x > 0 else '#2ecc71' for x in importance_df['Impact']]
    sns.barplot(x='Impact', y='Feature', data=importance_df, palette=colors)
    plt.title('Explainable AI: Top 10 Drivers of Customer Churn', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Impact Weight (Red = Drives Churn, Green = Prevents Churn)', fontsize=12)
    plt.ylabel('')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/1_Feature_Importance.png', dpi=300)
    plt.close()

    # ==========================================
    # GRAPH 2: Risk Probability Distribution
    # ==========================================
    print("🌊 Generating Risk Distribution KDE Plot...")
    plt.figure(figsize=(10, 6))
    
    sns.kdeplot(y_probs[y_test == 0], fill=True, color='#2ecc71', label='Loyal Customers', alpha=0.5, linewidth=2)
    sns.kdeplot(y_probs[y_test == 1], fill=True, color='#e74c3c', label='Churned Customers', alpha=0.5, linewidth=2)
    
    plt.axvline(x=0.30, color='black', linestyle='--', linewidth=2, label='Cost-Sensitive Threshold (30%)')
    
    plt.title('AI Risk Assessment: Probability Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Churn Probability', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/2_Risk_Distribution.png', dpi=300)
    plt.close()

    # ==========================================
    # GRAPH 3: Cost-Sensitive Confusion Matrix
    # ==========================================
    print("🧮 Generating Confusion Matrix Heatmap...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Predicted Loyal', 'Predicted Churn'], 
                yticklabels=['Actual Loyal', 'Actual Churn'],
                annot_kws={"size": 16, "weight": "bold"})
    
    plt.title('Model Performance (30% Risk Threshold)', fontsize=16, fontweight='bold', pad=20)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/3_Confusion_Matrix.png', dpi=300)
    plt.close()

    print(f"✅ All graphics successfully saved to the '{OUTPUT_DIR}' folder!")

if __name__ == "__main__":
    generate_graphs()