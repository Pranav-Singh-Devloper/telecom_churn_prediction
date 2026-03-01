import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Paths
DATA_PATH = 'data/raw/telecom_churn.csv'
MODEL_SAVE_PATH = 'models/churn_model.pkl'

def train_pipeline():
    # --- Load Data ---
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print("❌ Data file missing. Run data_loader.py to check.")
        return

    # --- Preprocessing Specific to Kaggle Dataset ---
    
    # 1. Clean Column Names (remove spaces to make them easier to use)
    # "Total day minutes" -> "Total_day_minutes"
    df.columns = df.columns.str.replace(' ', '_')
    
    # 2. Convert Target 'Churn' from True/False to 1/0
    # The dataset uses boolean True/False for churn
    df['Churn'] = df['Churn'].astype(int)

    # 3. Define Features (X) and Target (y)
    # We drop 'State' and 'Area_code' (simplification) and the Target 'Churn'
    drop_cols = ['State', 'Area_code', 'Churn']
    X = df.drop(columns=drop_cols)
    y = df['Churn']

    # --- Identify Column Types for the Pipeline ---
    # Categorical: 'International_plan', 'Voice_mail_plan'
    categorical_features = ['International_plan', 'Voice_mail_plan']
    
    # Numerical: Everything else (Account length, minutes, calls, charges, etc.)
    numeric_features = [col for col in X.columns if col not in categorical_features]

    # --- Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Build the Transformer ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    # --- Build the Pipeline ---
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000)) # Increased max_iter for convergence
    ])

    # --- Train ---
    print("⚙️ Training model on Kaggle dataset...")
    clf.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = clf.predict(X_test)
    print("\n📊 Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nDetailed Report:\n", classification_report(y_test, y_pred))

    # --- Save ---
    import os
    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_pipeline()