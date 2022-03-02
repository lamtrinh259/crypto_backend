from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pandas as pd
import joblib
from crypto_backend import get_data

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"Welcome": "To the world of cryptocurrency forecasting"}

@app.get("/SARIMAX_predict")
def predict(selected_crypto):
    """Return the next 30-day price forecast for the selected crypto

    Params: user-selected cryptocurrency
    """
    # Get the entire dataset from Kaggle (CLI)
    # kaggle datasets download -d tencars/392-crypto-currency-pairs-at-minute-resolution/

    # Get data
    df = get_data(selected_crypto)

    # Get the SARIMAX model
    SARIMAX = joblib.load('SARIMAX.joblib')

    # Predict the prices
    y_pred = SARIMAX.predict(df)
    # print(y_pred)
    return {'fare': y_pred[0]} #json_response

# For testing
