import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Import our custom cleaner!
from preprocessing import TelecomDataCleaner

DATA_PATH = 'data/raw/telecom_churn.csv'
MODEL_SAVE_PATH = 'models/churn_model.pkl'

def train_pipeline():
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print("❌ Data file missing.")
        return

    # Extract Target (We handle the True/False to 1/0 conversion here)
    y = df['Churn'].astype(int)
    X = df.drop(columns=['Churn'])

    # Define feature types (using the cleaned names with underscores)
    categorical_features = ['International_plan', 'Voice_mail_plan']
    
    # We list the exact numerical columns we want to use to avoid confusion
    numeric_features = [
        'Account_length', 'Number_vmail_messages', 'Total_day_minutes', 
        'Total_day_calls', 'Total_day_charge', 'Total_eve_minutes', 
        'Total_eve_calls', 'Total_eve_charge', 'Total_night_minutes', 
        'Total_night_calls', 'Total_night_charge', 'Total_intl_minutes', 
        'Total_intl_calls', 'Total_intl_charge', 'Customer_service_calls'
    ]

    # Build the Standard Scaler/Encoder
    column_transformer = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # The Ultimate Pipeline: Cleaner -> Transformer -> Model
    clf = Pipeline(steps=[
        ('cleaner', TelecomDataCleaner()),          # 1. Cleans column names & drops state/area
        ('preprocessor', column_transformer),       # 2. Scales numbers & encodes categories
        ('classifier', LogisticRegression(max_iter=1000)) # 3. Predicts churn
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("⚙️ Training model...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_pipeline()