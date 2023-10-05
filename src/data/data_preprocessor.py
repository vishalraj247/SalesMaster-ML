import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.features.feature_engineer import FeatureEngineer, MultiColumnLabelEncoder
import joblib

class DataPreparation:
    def __init__(self, sales_train=None, calendar=None, calendar_events=None, sell_prices=None):
        self.sales_train = sales_train
        self.calendar = calendar
        self.calendar_events = calendar_events
        self.sell_prices = sell_prices
        self.numerical_features = ['day_of_week', 'month', 'year', 'event_type_encoded']
        self.categorical_features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    
    def merge_data(self):
        """
        Merge sales_train with calendar, sell_prices, and calendar_events
        """
        if self.sales_train is None:
            raise ValueError("sales_train is None. It must be provided for merge_data operation.")
        # Extract day columns
        day_columns = [col for col in self.sales_train.columns if 'd_' in col]

        # Melt sales_train data
        sales_long = self.sales_train.melt(id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 
                                        value_vars=day_columns, 
                                        var_name='d', 
                                        value_name='sales')
        print(f"After melting sales_train: {sales_long.shape}")
        
        # Merge with calendar
        sales_long = pd.merge(sales_long, self.calendar, on='d', how='left')
        print(f"After merging with calendar: {sales_long.shape}")
        
        # Merge with sell prices
        merged_with_prices = pd.merge(sales_long, self.sell_prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
        print(f"After merging with sell prices: {merged_with_prices.shape}")
        # Fill missing values in 'sell_price' with 0
        merged_with_prices['sell_price'].fillna(0, inplace=True)

        # Handle aggregated calendar_events (if necessary)
        if self.calendar_events is not None:
            # Aggregate calendar_events by date
            agg_calendar_events = self.calendar_events.groupby('date').agg({
                'event_name': lambda x: ', '.join(x),
                'event_type': lambda x: ', '.join(x)
            }).reset_index()
            
            # Merge with aggregated calendar_events
            final_merged_data = pd.merge(merged_with_prices, agg_calendar_events, on='date', how='left')
            print(f"After merging with aggregated calendar_events: {final_merged_data.shape}")
        else:
            final_merged_data = merged_with_prices

        return final_merged_data
    
    def data_transformation(self, X):
        # Initialize and test numerical_transformer
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('scaler', StandardScaler())
        ])
        print("Numerical Transformer Output:")
        print(numerical_transformer.fit_transform(X[self.numerical_features]))

        # Initialize and test categorical_transformer
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('labelencoder', MultiColumnLabelEncoder())  # Using a custom transformer for multi-column Label Encoding
        ])
        print("Categorical Transformer Output:")
        print(categorical_transformer.fit_transform(X[self.categorical_features]))

        # Test ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, [col for col in self.numerical_features if col not in ['revenue', 'sales', 'sell_price']]),  # Exclude 'revenue', 'sales', and 'sell_price' as they are targets or should be excluded
                ('cat', categorical_transformer, self.categorical_features)
            ])
        print("Column Transformer Output:")
        print(preprocessor.fit_transform(X))
        
        return preprocessor

    def prepare_data(self):
        if self.sales_train is None or self.sell_prices is None:
            raise ValueError("sales_train and sell_prices must be provided for prepare_data operation.")
        # Step 1: Merge data
        merged_data = self.merge_data()
        
        # Step 2: Feature Engineering
        feature_engineer = FeatureEngineer()
        feature_engineer.fit(merged_data)
        feature_engineer.save_encoder('models/preprocessor_and_encoder/encoder.joblib')
        data_with_features = feature_engineer.fit_transform(merged_data)
        print(data_with_features.head())
        
        # Step 3: Split data into features (X) and target (y)
        X = data_with_features.drop('revenue', axis=1)
        y = data_with_features['revenue']
        
        print(f"Data type of X before fit_transform: {type(X)}")
        print(f"Shape of X before fit_transform: {X.shape}")

        # Step 4: Data Transformation
        self.preprocessor = self.data_transformation(X)
        X_transformed = self.preprocessor.fit_transform(X)
        joblib.dump(self.preprocessor, 'models/preprocessor_and_encoder/preprocessor.joblib')

        print(f"Shape of X_transformed before DataFrame conversion: {X_transformed.shape}")
        
        # DEBUGGING: Print the shapes and types of data at various steps
        print(f"Type of X_transformed: {type(X_transformed)}")
        print(f"Shape of X_transformed: {X_transformed.shape}")

        # If X_transformed is a sparse matrix, we may need to convert it to a dense array
        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()
        
        # Creating the columns list
        numerical_features_transformed = [col for col in self.numerical_features if col != 'revenue']
        columns = numerical_features_transformed + self.categorical_features
        print(f"Number of columns: {len(columns)}")

        # DEBUGGING: Print the columns list
        print(f"Columns: {columns}")

        # Convert to DataFrame
        X_transformed = pd.DataFrame(X_transformed, columns=columns)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=0)
        
        return X_train, X_test, y_train, y_test, merged_data, data_with_features, X_transformed

    def prepare_prophet_data(self):
        """
        Prepares and aggregates data for Prophet forecasting.
        This method groups the data by date, calculates total revenue per day,
        and renames the columns to be compatible with Prophet.
        
        Returns:
            pd.DataFrame: A DataFrame with 'ds' and 'y' columns ready for Prophet forecasting.
        """
        if self.sales_train is None or self.sell_prices is None:
            raise ValueError("sales_train and sell_prices must be provided for prepare_prophet_data operation.")
        # Call the merge_data method to get the merged dataset
        merged_data = self.merge_data()

        # Calculate daily revenue
        merged_data['revenue'] = merged_data['sell_price'] * merged_data['sales']
        
        # Aggregate revenue per day
        daily_sales = merged_data.groupby('date')['revenue'].sum().reset_index()
        
        # Ensure 'date' is in the datetime format
        daily_sales['date'] = pd.to_datetime(daily_sales['date'])
        
        # Count number of unique events per day
        events_per_day = merged_data.groupby('date')['event_name'].nunique().reset_index()
        events_per_day.columns = ['date', 'num_events']
        
        # Ensure 'date' is in the datetime format for events_per_day
        events_per_day['date'] = pd.to_datetime(events_per_day['date'])
        
        # Merge with daily_sales to add 'num_events' column
        daily_sales = pd.merge(daily_sales, events_per_day, on='date', how='left')
        
        # Fill NaN values in 'num_events' with 0 (days without events)
        daily_sales['num_events'].fillna(0, inplace=True)
        
        # Rename columns for compatibility with Prophet
        daily_sales.columns = ['ds', 'y', 'num_events']
        
        return daily_sales, merged_data

    def split_time_series(self, data, test_size=0.2):
        """
        Split time series data into training and test sets.
        """
        split_idx = int(len(data) * (1 - test_size))
        train = data[:split_idx]
        test = data[split_idx:]
        print(f"Splitting at index: {split_idx}")
        print(f"Train data range: {train['ds'].min()} - {train['ds'].max()}")
        print(f"Test data range: {test['ds'].min()} - {test['ds'].max()}")
        return train, test, split_idx
    
    def load_preprocessor(self, filepath):
        self.preprocessor = joblib.load(filepath)
        print(f"Preprocessor loaded from {filepath}")

    def prepare_single_data_point(self, item_id, store_id, date):
        """
        Prepare a single data point for prediction.

        Parameters:
        item_id (str): ID of the item.
        store_id (str): ID of the store.
        date (str): Date string.

        Returns:
        pd.DataFrame: Prepared data point for prediction.
        """
        # Step 1: Create a DataFrame with the single data point
        X_single = pd.DataFrame({'item_id': [item_id], 'store_id': [store_id], 'date': [date]})

        # Extracting dept_id and cat_id from item_id
        X_single['dept_id'] = X_single['item_id'].apply(lambda x: x.rsplit('_', 1)[0])  # This will take the substring before the last underscore
        X_single['cat_id'] = X_single['dept_id'].apply(lambda x: x.split('_')[0])  # This will take the substring before the first underscore in dept_id

        # Extracting state_id from store_id
        X_single['state_id'] = X_single['store_id'].apply(lambda x: x.split('_')[0])  # This will take the substring before the first underscore

        # Convert 'date' to datetime before merging
        X_single['date'] = pd.to_datetime(X_single['date'])
        # Ensure 'date' in self.calendar is also in datetime format
        self.calendar['date'] = pd.to_datetime(self.calendar['date'])

        # Step 2: Merge with calendar
        print("Shape before calendar merge:", X_single.shape)
        X_single = pd.merge(X_single, self.calendar, left_on='date', right_on='date', how='left')
        print("Shape after calendar merge:", X_single.shape)

        # Merge with aggregated calendar_events if available
        if self.calendar_events is not None:
            agg_calendar_events = self.calendar_events.groupby('date').agg({
                'event_name': lambda x: ', '.join(x),
                'event_type': lambda x: ', '.join(x)
            }).reset_index()
            
            # Ensure 'date' in agg_calendar_events is also in datetime format
            agg_calendar_events['date'] = pd.to_datetime(agg_calendar_events['date'])

            # Merge with aggregated calendar_events
            print("Shape before calendar_events merge:", X_single.shape)
            X_single = pd.merge(X_single, agg_calendar_events, on='date', how='left')
            print("Shape after calendar_events merge:", X_single.shape)
        
        # Handle NaNs for event_name and event_type
        X_single.fillna({'event_name': 'NoEvent', 'event_type': 'NoType'}, inplace=True)

        # Step 3: Feature Engineering
        feature_engineer = FeatureEngineer()
        feature_engineer.load_encoder('models/preprocessor_and_encoder/encoder.joblib') # Loading the fitted encoder
        X_single = feature_engineer.process_single_data_point(X_single)

        # Step 4: Data Transformation
        # Using the preprocessor fitted on the training data for transformation
        # Ensure the columns order matches the training data
        self.load_preprocessor('models/preprocessor_and_encoder/preprocessor.joblib')

        columns_after_transformation = ['day_of_week', 'month', 'year', 'event_type_encoded', 'item_id',
                                        'dept_id', 'cat_id', 'store_id','state_id']

        X_transformed_single = self.preprocessor.transform(X_single)
        
        X_transformed_single = pd.DataFrame(X_transformed_single, columns=columns_after_transformation)

        return X_transformed_single
