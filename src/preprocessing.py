import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TelecomDataCleaner(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer to clean the Kaggle Telecom dataset.
    This ensures both training data and live UI data are processed identically.
    """
    def __init__(self):
        # We define the columns we want to drop here
        self.cols_to_drop = ['State', 'Area_code']

    def fit(self, X, y=None):
        # Fit does nothing here, but is required by Scikit-Learn
        return self

    def transform(self, X):
        # Always work on a copy to avoid modifying the original data
        X_copy = X.copy()
        
        # 1. Clean Column Names (replace spaces with underscores)
        if isinstance(X_copy, pd.DataFrame):
            X_copy.columns = X_copy.columns.str.replace(' ', '_')
            
            # 2. Drop unnecessary columns (if they exist in the input)
            existing_cols_to_drop = [c for c in self.cols_to_drop if c in X_copy.columns]
            X_copy = X_copy.drop(columns=existing_cols_to_drop)
            
        return X_copy