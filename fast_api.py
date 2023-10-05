from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime
from src.data.make_dataset import load_data_prediction
from src.data.data_preprocessor import DataPreparation
from src.features.feature_engineer import FeatureEngineer
from src.models.prophet_forecaster import ProphetForecaster

# Load Models
xgboost_model = joblib.load("models/predictive/final_xgboost_model.joblib")
prophet_model = joblib.load("models/forecasting/final_prophet_model.joblib")

# Initialize FastAPI app
app = FastAPI()

# Load data
calendar, calendar_events = load_data_prediction()

# Load the data preparation and feature engineering instances or functions
# Initialize DataPreparation with the necessary datasets only
data_prep = DataPreparation(calendar=calendar, calendar_events=calendar_events)
feature_engineer = FeatureEngineer()
forecaster = ProphetForecaster()

@app.get("/")
async def read_root():
    return {
    "description": "SalesMaster-ML API: An advanced sales forecasting system.",
    "project_objectives": "To accurately forecast sales volume at national level and for specific store-item combinations, aiding in efficient inventory management and strategic planning.",
    "endpoints": {
        "/": {
            "description": "Provides a brief overview of the SalesMaster-ML API including its objectives, list of endpoints, expected input parameters, and output formats.",
            "method": "GET"
        },
        "/health/": {
            "description": "Endpoint to verify that the API is running and healthy. Returns a status code of 200 along with a custom welcome message.",
            "method": "GET"
        },
        "/sales/national/": {
            "description": "Endpoint for national sales forecast. Returns the predicted sales volume for the next 7 days based on the input date and number of events.",
            "method": "GET",
            "parameters": {
                "date": "The start date for the forecast period in 'YYYY-MM-DD' format.",
                "num_events": "Estimated number of events occurring in the forecast period (float)."
            }
        },
        "/sales/stores/items/": {
            "description": "Endpoint for item-specific sales forecast in a given store. Returns the predicted sales volume for the specified item, store, and date.",
            "method": "GET",
            "parameters": {
                "item_id": "The unique identifier for the item.",
                "store_id": "The unique identifier for the store.",
                "date": "The date for which the sales forecast is requested in 'YYYY-MM-DD' format."
            }
        }
    },
    "github_repository": "https://github.com/vishalraj247/SalesMaster-ML.git"
}

@app.get("/health/")
async def read_health():
    return {"status": 200, "message": "Welcome to SalesMaster-ML API! The API is healthy and ready to forecast sales!"}

@app.get("/sales/national/")
async def get_national_sales(date: str, num_events: float):
    try:
        # Convert string date to datetime object
        start_date = datetime.strptime(date, '%Y-%m-%d')

        # Increment start_date by one day
        start_date += pd.Timedelta(days=1)

        # Create a dataframe with date range for the next 7 days
        future_dates = pd.date_range(start=start_date, periods=7).to_frame(index=False, name='ds')

        # Assign the estimated number of events to future dates
        future_dates['num_events'] = num_events

        # Make prediction using loaded Prophet model
        forecast = prophet_model.predict(future_dates)

        # Extract predictions for the next 7 days
        predictions = forecast[['ds', 'yhat']].set_index('ds').to_dict()['yhat']

        # Convert date keys from Timestamp to string in the response
        str_predictions = {str(key): value for key, value in predictions.items()}

        # Return the predictions
        return {"predictions": str_predictions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/sales/stores/items/")
async def get_item_sales(item_id: str, store_id: str, date: str):
    try:
        # Prepare single data point using the provided request data
        prepared_data = data_prep.prepare_single_data_point(item_id, store_id, date)

        # Make prediction using XGBoost model
        prediction = xgboost_model.predict(prepared_data)

        # Return the prediction
        return {"prediction": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))