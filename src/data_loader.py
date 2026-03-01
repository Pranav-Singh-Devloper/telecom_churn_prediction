import pandas as pd
import os

def load_data():
    """
    Loads the Kaggle Telecom Churn dataset and performs basic validation.
    """
    # Define the path relative to this script
    file_path = 'data/raw/telecom_churn.csv'
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        print("   Please download 'churn-bigml-80.csv' from Kaggle,")
        print("   rename it to 'telecom_churn.csv', and place it in 'data/raw/'.")
        return None
    
    df = pd.read_csv(file_path)
    
    # Check if we have the expected columns
    expected_col = "Total day minutes"
    if expected_col not in df.columns:
        print(f"❌ Error: The file doesn't look right. Missing column '{expected_col}'")
        return None
        
    print(f"✅ Data loaded successfully! Shape: {df.shape}")
    print("Columns found:", df.columns.tolist())
    return df

if __name__ == "__main__":
    load_data()