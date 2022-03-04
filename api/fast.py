from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pandas as pd
import joblib
from crypto_backend import data
from crypto_backend.trainer import Trainer

app = FastAPI()
cache = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Logic runing all the models



@app.get("/")
def index():
    return {"Welcome": "To the world of cryptocurrency forecasting"}

@app.get("/SARIMAX_predict")
def predict_sarimax(selected_crypto):
    """Return the next 14-day price forecast for the selected crypto

    Params: user-selected cryptocurrency
    """
    # Get the entire dataset from Kaggle (CLI)
    # kaggle datasets download -d tencars/392-crypto-currency-pairs-at-minute-resolution/

    # Get data
    df = data.get_data(selected_crypto)

    # Get the SARIMAX model
    SARIMAX = joblib.load('SARIMAX.joblib')

    # Predict the prices
    y_pred = SARIMAX.predict(df)
    # print(y_pred)
    return {'fare': y_pred[0]} #json_response

@app.get("/fbprophet_predict")
def predict_fb(selected_crypto):
    """
    Returns 14 day prediction
    """
    if 'fb_prophet' in cache and selected_crypto in cache['fb_prophet']:
        return cache['fb_prophet'][selected_crypto]

    trainer = Trainer(selected_crypto)
    trainer.load_data()
    result = trainer.prophecy_predict()
    cache['fb_prophet'] = {selected_crypto : result}

    return cache['fb_prophet'][selected_crypto]
