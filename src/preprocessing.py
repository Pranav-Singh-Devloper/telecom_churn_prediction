import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TelecomDataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cols_to_drop = ['State', 'Area code', 'Phone number']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # --- FEATURE ENGINEERING ---
        # 1. Service Intensity: Total calls / Account Length
        # (High intensity often correlates with high engagement OR high friction)
        X['Service_Intensity'] = (X['Total day calls'] + X['Total eve calls']) / (X['Account length'] + 1)
        
        # 2. Cost Per Minute: Total Charge / Total Minutes
        # (Helps identify if a customer feels overcharged)
        X['Day_Cost_Per_Min'] = X['Total day charge'] / (X['Total day minutes'] + 1)
        
        # 3. Support Dependency: Customer Service Calls / Total Calls
        # (High ratio is a massive churn indicator)
        X['Support_Friction'] = X['Customer service calls'] / (X['Total day calls'] + 1)

        # Drop unnecessary columns
        X = X.drop(columns=self.cols_to_drop, errors='ignore')
        
        # Standardize column names for the model
        X.columns = [c.replace(' ', '_') for c in X.columns]
        return X

    def get_feature_names_out(self, input_features=None):
        # Update names to match the engineered features
        base_features = [f for f in input_features if f not in self.cols_to_drop]
        engineered = ['Service_Intensity', 'Day_Cost_Per_Min', 'Support_Friction']
        return [f.replace(' ', '_') for f in base_features] + engineered