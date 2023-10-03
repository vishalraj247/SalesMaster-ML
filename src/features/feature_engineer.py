
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

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
