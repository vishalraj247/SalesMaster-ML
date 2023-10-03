import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import joblib

class XGBoostRegressor:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def objective(self, space):
        model = xgb.XGBRegressor(
            n_estimators =space['n_estimators'], 
            max_depth = int(space['max_depth']),
            min_child_weight = space['min_child_weight'],
            subsample = space['subsample'],
            learning_rate = space['learning_rate'],
            gamma = space['gamma'],
            colsample_bytree = space['colsample_bytree']
        )
        
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_val)
        mse = mean_squared_error(self.y_val, y_pred)
        
        return {'loss': mse, 'status': STATUS_OK}

    def optimize(self):
        space ={
            'max_depth': hp.quniform('max_depth', 5, 15, 1),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'n_estimators': hp.choice('n_estimators', range(20, 205, 5)),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
            'gamma': hp.uniform('gamma', 0, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1)
        }
        
        trials = Trials()
        best = fmin(fn=self.objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=50,
                    trials=trials)
        
        return best
    
    def train_final_model(self, best_params):
        """
        Train the model with the best hyperparameters found by HyperOpt
        """
        model = xgb.XGBRegressor(
            n_estimators =best_params['n_estimators'], 
            max_depth = int(best_params['max_depth']),
            min_child_weight = best_params['min_child_weight'],
            subsample = best_params['subsample'],
            learning_rate = best_params['learning_rate'],
            gamma = best_params['gamma'],
            colsample_bytree = best_params['colsample_bytree']
        )
        
        model.fit(self.X_train, self.y_train)
        return model
    
    def save_model(self, model, filepath):
        """
        Save the trained model to a file and return success message
        """
        try:
            joblib.dump(model, filepath)
            return f"Model saved successfully at {filepath}"
        except Exception as e:
            return f"An error occurred while saving the model: {e}"

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the model on the test set and print the results
        """
        # Predictions
        y_pred = model.predict(X_test)

        # Evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # Print results
        print(f'Model Performance on Test Set:')
        print(f'MAE: {mae}')
        print(f'MSE: {mse}')
        print(f'RMSE: {rmse}')

        return mae, mse, rmse