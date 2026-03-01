import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# SOTA Addition 1: Imbalanced-Learn Pipeline & SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from preprocessing import TelecomDataCleaner

DATA_PATH = 'data/raw/telecom_churn.csv'
MODEL_SAVE_PATH = 'models/churn_model.pkl'

def train_pipeline():
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print("❌ Data file missing.")
        return

    y = df['Churn'].astype(int)
    X = df.drop(columns=['Churn'])

    categorical_features = ['International_plan', 'Voice_mail_plan']
    numeric_features = [
        'Account_length', 'Number_vmail_messages', 'Total_day_minutes', 
        'Total_day_calls', 'Total_day_charge', 'Total_eve_minutes', 
        'Total_eve_calls', 'Total_eve_charge', 'Total_night_minutes', 
        'Total_night_calls', 'Total_night_charge', 'Total_intl_minutes', 
        'Total_intl_calls', 'Total_intl_charge', 'Customer_service_calls',
        'Service_Intensity', 'Day_Cost_Per_Min', 'Support_Friction'
    ]

    column_transformer = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # --- SOTA Upgrade: The Intelligent Pipeline ---
    # We use ImbPipeline so SMOTE only generates data during training, not testing!
    clf = ImbPipeline(steps=[
        ('cleaner', TelecomDataCleaner()),
        ('preprocessor', column_transformer),
        ('smote', SMOTE(random_state=42)), # Synthesizes minority churners
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000)) # Cost-sensitive penalization
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("⚙️ Training model with SMOTE & Cost-Sensitive Learning...")
    clf.fit(X_train, y_train)

    # --- SOTA Addition 2: Cost-Sensitive Threshold Tuning ---
    # Instead of predict(), we use predict_proba() and set a custom threshold of 30%
    print("\n📊 Evaluating with Custom Risk Threshold (30%)...")
    y_probs = clf.predict_proba(X_test)[:, 1]
    CUSTOM_THRESHOLD = 0.30
    y_pred_tuned = (y_probs >= CUSTOM_THRESHOLD).astype(int)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred_tuned):.2f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tuned))
    print("\nDetailed Report:\n", classification_report(y_test, y_pred_tuned))

    # --- SOTA Addition 3: Explainable AI (XAI) ---
    print("\n🧠 Explainable AI: Top Churn Drivers")
    # 1. Extract the feature names after they pass through the transformer
    feature_names = clf.named_steps['preprocessor'].get_feature_names_out()
    
    # 2. Extract the weights (coefficients) from the brain of the Logistic Regression
    coefficients = clf.named_steps['classifier'].coef_[0]
    
    # 3. Zip them together and sort by absolute impact
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Impact_Weight': coefficients
    })
    # Sort by absolute value to see the strongest drivers (positive or negative)
    feature_importance['Absolute_Impact'] = feature_importance['Impact_Weight'].abs()
    feature_importance = feature_importance.sort_values(by='Absolute_Impact', ascending=False)
    
    print("The higher the positive weight, the more it DRIVES churn.")
    print("The lower the negative weight, the more it PREVENTS churn.\n")
    print(feature_importance[['Feature', 'Impact_Weight']].head(5).to_string(index=False))

    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, MODEL_SAVE_PATH)
    print(f"\n✅ Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_pipeline()