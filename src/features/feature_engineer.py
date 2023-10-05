
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import joblib

class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            output = X.copy()
        else:  # Handle NumPy arrays
            output = pd.DataFrame(X)
        
        for col in output.columns:
            output[col] = LabelEncoder().fit_transform(output[col].astype(str))
        return output.values  # Return NumPy array, not DataFrame

class FeatureEngineer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.encoder = LabelEncoder()
        
    def fit(self, X, y=None):
        event_types = X['event_type'].fillna('NoType')
        self.encoder.fit(event_types)
        return self
    
    def transform(self, X):
        # Create revenue feature
        X['revenue'] = X['sell_price'] * X['sales']
        
        # Extract date features
        X['date'] = pd.to_datetime(X['date'])
        X['day_of_week'] = X['date'].dt.dayofweek
        X['month'] = X['date'].dt.month
        X['year'] = X['date'].dt.year
        
        # Handle events with placeholder for NaNs
        X = X.fillna({'event_name': 'NoEvent', 'event_type': 'NoType'})
        X['event'] = X['event_name'].apply(lambda x: 0 if x == 'NoEvent' else 1)  # Binary column indicating an event
        
        # Label Encoding for event_type
        X['event_type_encoded'] = self.encoder.transform(X['event_type'])
        
        # Drop the original event_type and event_name columns
        X = X.drop(['event_type', 'event_name'], axis=1)
        
        return X
    
    def save_encoder(self, filepath):
        joblib.dump(self.encoder, filepath)
        print(f"Encoder saved at {filepath}")

    def load_encoder(self, filepath):
        self.encoder = joblib.load(filepath)
        print(f"Encoder loaded from {filepath}")
    
    def process_single_data_point(self, X_single):
        """
        Process a single data point or small batch of data.
        """
        # Handle X_single data point or small batch in the same way we handled X in the transform method

        # Example: Adjust for single data point
        if 'sell_price' not in X_single.columns:
            X_single['sell_price'] = 0  # or some default value
            
        X_single['date'] = pd.to_datetime(X_single['date'])
        X_single['day_of_week'] = X_single['date'].dt.dayofweek
        X_single['month'] = X_single['date'].dt.month
        X_single['year'] = X_single['date'].dt.year
        X_single = X_single.fillna({'event_name': 'NoEvent', 'event_type': 'NoType'})
        X_single['event'] = X_single['event_name'].apply(lambda x: 0 if x == 'NoEvent' else 1)
        X_single['event_type_encoded'] = self.encoder.transform(X_single['event_type'])
        X_single = X_single.drop(['event_type', 'event_name'], axis=1)

        return X_single
