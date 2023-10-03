import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from feature_engineer import FeatureEngineer, MultiColumnLabelEncoder

class DataPreparation:
    def __init__(self, sales_train, calendar, calendar_events, sell_prices):
        self.sales_train = sales_train
        self.calendar = calendar
        self.calendar_events = calendar_events
        self.sell_prices = sell_prices
        self.numerical_features = ['sell_price', 'sales', 'day_of_week', 'month', 'year', 'event_type_encoded']
        self.categorical_features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    
    def merge_data(self):
        """
        Merge sales_train with calendar, sell_prices, and calendar_events
        """
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
                ('num', numerical_transformer, [col for col in self.numerical_features if col != 'revenue']),  # Exclude 'revenue' as it's the target
                ('cat', categorical_transformer, self.categorical_features)
            ])
        print("Column Transformer Output:")
        print(preprocessor.fit_transform(X))
        
        return preprocessor

    def prepare_data(self):
        # Step 1: Merge data
        merged_data = self.merge_data()
        
        # Step 2: Feature Engineering
        feature_engineer = FeatureEngineer()
        data_with_features = feature_engineer.fit_transform(merged_data)
        print(data_with_features.head())
        
        # Step 3: Split data into features (X) and target (y)
        X = data_with_features.drop('revenue', axis=1)
        y = data_with_features['revenue']
        
        print(f"Data type of X before fit_transform: {type(X)}")
        print(f"Shape of X before fit_transform: {X.shape}")

        # Step 4: Data Transformation
        preprocessor = self.data_transformation(X)
        X_transformed = preprocessor.fit_transform(X)

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
        Prepare data for forecasting with Prophet.
        """
        # Call the merge_data method and store its return value
        merged_data = self.merge_data()

         # Create revenue column here
        merged_data['revenue'] = merged_data['sell_price'] * merged_data['sales']
        
        # Aggregate total sales per day
        daily_sales = merged_data.groupby('date')['revenue'].sum().reset_index()
        
        # Rename columns for Prophet compatibility
        daily_sales.columns = ['ds', 'y']
        return daily_sales