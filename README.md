# SalesMaster-ML

## Overview
SalesMaster-ML is a robust system designed for sales prediction and forecasting in the retail sector, employing advanced machine learning models.

## Structure
Below is the project's structure:

Project Organization
------------
```
SalesMaster-ML/
│
├── Dockerfile
├── .dockerignore
├── docs/
├── models/
│   ├── forecasting/
│   │   └── final_prophet_model.joblib
│   ├── predictive/
│   │   └── final_xgboost_model.joblib
│   └── preprocessor and encoder/
│       ├── encoder.joblib
│       └── preprocessor.joblib
├── notebooks/
│   ├── forecasting/
│   │   └── raj_vishal-14227627-forecasting_prophet.ipynb
│   └── predictive/
│       └── raj_vishal-14227627-predictive_xgboost.ipynb
├── references/
├── reports/
│   ├── figures/
│   │   ├── Actual vs Forecasted Values.png
│   │   └── Actual vs Forecasted Values(data points).png
│   ├── Vishal_Raj-14227627-Experiment_Report.pdf
│   └── VishalRaj_14227627_AT2FinalReport.pdf
├── src/
    ├── data/
    │   ├── data_preprocessor.py
    │   └── make_dataset.py
    ├── features/
    │   └── feature_engineer.py
    ├── models/
    │   ├── prophet_forecaster.py
    │   └── xgboost_predict.py
├── fast_api.py
├── LICENSE
├── Makefile
├── Procfile
├── requirements.txt
├── runtime.txt
├── setup.py
├── test_environment.py
└── tox.ini
```

## Getting Started
1. Clone this repository
2. Navigate to the project directory
3. Install dependencies: `pip install -r requirements.txt`

## Deploying to Heroku
### Prerequisites
- Heroku account
- Heroku CLI installed

### Steps
1. Login to Heroku: `heroku login`
2. Create a new Heroku app: `heroku create your-app-name`
3. Set Heroku remote: `git remote add heroku your-heroku-git-url`
4. Push to Heroku: `git push heroku master`
5. Open your app with: `heroku open`

## Usage
Refer to the API documentation: [API Docs](https://sales-master-app-031d89e0c0e1.herokuapp.com/docs#/default/read_root__get)

## License
This project is licensed under the terms of the MIT license.

## Acknowledgments
- Prophet and XGBoost for providing robust algorithms for forecasting and prediction respectively.
- FastAPI for providing an efficient way to deploy ML models.
