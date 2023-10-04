from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import matplotlib.pyplot as plt
import numpy as np

class ProphetForecaster:
    def __init__(self):
        pass
    
    def train_model(self, data):
        # Initialize a Prophet object with daily and yearly seasonality
        model = Prophet(yearly_seasonality=True, daily_seasonality=True)
        
        # Add 'num_events' as an additional regressor to the model
        model.add_regressor('num_events')
        
        # Fit the model to the data
        model.fit(data)
        return model
    
    def forecast(self, model, days, future_regressors=None):
        # Create a DataFrame with future dates
        future = model.make_future_dataframe(periods=days)
        
        # If future regressor values are provided, add them to the 'future' DataFrame
        if future_regressors is not None:
            future['num_events'] = future_regressors
        
        # Generate a forecast for the future dates
        forecast = model.predict(future)
        return forecast
    
    def evaluate_forecast(self, y_true, forecast):
        """
        Calculate MAPE and SMAPE for the forecast.
        """
        forecasted = forecast['yhat'][-len(y_true):]
        mape = np.mean(np.abs((y_true - forecasted) / y_true)) * 100
        smape = np.mean(2 * np.abs(y_true - forecasted) / (np.abs(y_true) + np.abs(forecasted))) * 100

        print(f'MAPE: {mape:.2f}%')
        print(f'SMAPE: {smape:.2f}%')

    def plot_forecast(self, model, forecast, data):
        """
        Plot the Prophet forecast.
        """
        fig = model.plot(forecast)
        a = add_changepoints_to_plot(fig.gca(), model, forecast)
        plt.show()