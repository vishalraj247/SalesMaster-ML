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
        "API": {
            "name": "SalesMaster-ML API",
            "version": "1.0",
            "description": "An advanced sales forecasting system that accurately forecasts sales volume at the national level and for specific store-item combinations.",
            "repository": "https://github.com/vishalraj247/SalesMaster-ML.git"
        },
        "endpoints": {
            "root": {
                "path": "/",
                "method": "GET",
                "description": "Provides an overview of the SalesMaster-ML API.",
                "parameters": None
            },
            "health": {
                "path": "/health/",
                "method": "GET",
                "description": "Verifies that the API is running and healthy.",
                "parameters": None
            },
            "national_sales": {
                "path": "/sales/national/?date=YYYY-MM-DD&num_events=0",
                "method": "GET",
                "description": "Forecasts national sales for the next 7 days based on the input date and number of events which is 0 usually.",
                "parameters": {
                    "date": {
                        "description": "Start date for the forecast period.",
                        "type": "string",
                        "format": "YYYY-MM-DD"
                    },
                    "num_events": {
                        "description": "Estimated number of events in the forecast period.",
                        "type": "float"
                    }
                }
            },
            "item_sales": {
                "path": "/sales/stores/items/?item_id=ITEM123&store_id=STORE456&date=YYYY-MM-DD",
                "method": "GET",
                "description": "Forecasts sales for a specific item in a given store on a specified date.",
                "parameters": {
                    "item_id": {
                        "description": "Unique identifier for the item.",
                        "type": "string"
                    },
                    "store_id": {
                        "description": "Unique identifier for the store.",
                        "type": "string"
                    },
                    "date": {
                        "description": "Date for the sales forecast.",
                        "type": "string",
                        "format": "YYYY-MM-DD"
                    }
                }
            }
        }
    }

@app.get("/health/")
async def read_health():
    return {
        "status": {
            "code": 200,
            "type": "Success",
            "message": "OK"
        },
        "data": {
            "welcomeMessage": "Welcome to SalesMaster-ML API!",
            "healthStatus": "Healthy",
            "capabilities": "Ready to forecast sales!"
        }
    }

@app.get("/sales/national/")
async def get_national_sales(date: str, num_events: float):
    try:
        start_date = datetime.strptime(date, '%Y-%m-%d')
        start_date += pd.Timedelta(days=1)
        future_dates = pd.date_range(start=start_date, periods=7).to_frame(index=False, name='ds')
        future_dates['num_events'] = num_events

        forecast = prophet_model.predict(future_dates)
        predictions = forecast[['ds', 'yhat']].set_index('ds').to_dict()['yhat']
        str_predictions = {str(key): value for key, value in predictions.items()}

        return {
            "status": {
                "code": 200,
                "type": "Success",
                "message": "OK"
            },
            "data": {
                "forecastStartDate": date,
                "forecastEndDate": str(start_date + pd.Timedelta(days=6)),
                "predictions": str_predictions
            }
        }
    except Exception as e:
        return {
            "status": {
                "code": 400,
                "type": "Error",
                "message": str(e)
            },
            "data": None
        }

@app.get("/sales/stores/items/")
async def get_item_sales(item_id: str, store_id: str, date: str):
    try:
        prepared_data = data_prep.prepare_single_data_point(item_id, store_id, date)
        prediction = xgboost_model.predict(prepared_data)

        return {
            "status": {
                "code": 200,
                "type": "Success",
                "message": "OK"
            },
            "data": {
                "itemId": item_id,
                "storeId": store_id,
                "forecastDate": date,
                "predictedSales": float(prediction)
            }
        }
    except Exception as e:
        return {
            "status": {
                "code": 400,
                "type": "Error",
                "message": str(e)
            },
            "data": None
        }