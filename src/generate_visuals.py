import sys
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.dirname(__file__))
from preprocessing import TelecomDataCleaner

DATA_PATH = 'data/raw/telecom_churn.csv'
MODEL_PATH = 'models/churn_model.pkl'
OUTPUT_DIR = 'visualizations'

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("notebook", font_scale=1.2)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_graphs():
    print("📊 Loading data and model...")
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    # --- EDA 1: Correlation Heatmap ---
    print("🔍 Generating EDA: Correlation Heatmap...")
    plt.figure(figsize=(12, 10))
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('EDA: Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/4_Correlation_Heatmap.png', dpi=300)
    plt.close()

    # --- EDA 2: Churn vs Customer Service Calls ---
    print("🔍 Generating EDA: Service Calls Impact...")
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Customer service calls', hue='Churn', palette=['#2ecc71', '#e74c3c'])
    plt.title('EDA: Churn Rate by Customer Service Calls', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Number of Customer Service Calls')
    plt.ylabel('Number of Customers')
    plt.legend(title='Churn', labels=['Retained', 'Churned'])
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/5_Service_Calls_EDA.png', dpi=300)
    plt.close()

    # --- Metrics Preparation ---
    y = df['Churn'].astype(int)
    X = df.drop(columns=['Churn'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= 0.30).astype(int)

    # --- XAI & Evaluation Visuals (From Previous Step) ---
    print("📈 Generating Model Evaluation Visuals...")
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    coefficients = model.named_steps['classifier'].coef_[0]
    
    importance_df = pd.DataFrame({'Feature': feature_names, 'Impact': coefficients})
    importance_df['Absolute_Impact'] = importance_df['Impact'].abs()
    importance_df = importance_df.sort_values(by='Absolute_Impact', ascending=False).head(10)

    # 1. Feature Importance
    plt.figure(figsize=(10, 6))
    colors = ['#e74c3c' if x > 0 else '#2ecc71' for x in importance_df['Impact']]
    sns.barplot(x='Impact', y='Feature', data=importance_df, palette=colors)
    plt.title('Explainable AI: Top 10 Drivers of Customer Churn', fontsize=16, fontweight='bold', pad=20)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/1_Feature_Importance.png', dpi=300)
    plt.close()

    # 2. Risk Distribution
    plt.figure(figsize=(10, 6))
    sns.kdeplot(y_probs[y_test == 0], fill=True, color='#2ecc71', label='Loyal Customers', alpha=0.5)
    sns.kdeplot(y_probs[y_test == 1], fill=True, color='#e74c3c', label='Churned Customers', alpha=0.5)
    plt.axvline(x=0.30, color='black', linestyle='--', linewidth=2, label='Cost-Sensitive Threshold (30%)')
    plt.title('Evaluation: Probability Distribution & Threshold', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/2_Risk_Distribution.png', dpi=300)
    plt.close()

    # 3. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Predicted Loyal', 'Predicted Churn'], 
                yticklabels=['Actual Loyal', 'Actual Churn'],
                annot_kws={"size": 16, "weight": "bold"})
    plt.title('Evaluation: Confusion Matrix (30% Threshold)', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/3_Confusion_Matrix.png', dpi=300)
    plt.close()

    print(f"✅ All EDA and Evaluation graphics saved to '{OUTPUT_DIR}'!")

if __name__ == "__main__":
    generate_graphs()