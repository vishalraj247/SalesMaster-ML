from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from data_preprocessor import DataPreparation
from feature_engineer import FeatureEngineer
from prophet_forecaster import ProphetForecaster

forecaster = ProphetForecaster()
# Load Models
xgboost_model = joblib.load("../../models/predictive/final_xgboost_model.joblib")
prophet_model = forecaster.load_model("../../models/forecasting/final_prophet_model")

# Initialize FastAPI app
app = FastAPI()

# Create Pydantic models for request and response
class XGBoostRequest(BaseModel):
    item_id: str
    store_id: str
    date: str

class ProphetRequest(BaseModel):
    date: str
    num_events: float

class PredictionResponse(BaseModel):
    prediction: float

# Load the data preparation and feature engineering instances or functions
data_prep = DataPreparation(sales_train, calendar, calendar_events, sell_prices)
feature_engineer = FeatureEngineer()

# XGBoost Endpoint
@app.post("/predict_xgboost", response_model=PredictionResponse)
async def predict_xgboost(request: XGBoostRequest):
    try:
        # Prepare single data point using the provided request data
        prepared_data = data_prep.prepare_single_data_point(request.item_id, request.store_id, request.date)
        
        # Make prediction using XGBoost model
        prediction = xgboost_model.predict(prepared_data)

        # Return the prediction
        return PredictionResponse(prediction=float(prediction))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Prophet Endpoint
@app.post("/predict_prophet", response_model=PredictionResponse)
async def predict_prophet(request: ProphetRequest):
    try:
        # Prepare future dataframe for Prophet
        future = pd.DataFrame({"ds": [request.date], "num_events": [request.num_events]})
        
        # Make prediction using Prophet model
        forecast = prophet_model.predict(future)
        prediction = forecast['yhat'].iloc[0]

        # Return the prediction
        return PredictionResponse(prediction=float(prediction))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))