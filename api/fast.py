from genericpath import isfile
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

# Initialize Model for Cache
@app.on_event("startup")
def app_start():
    currency = [
        'BTC',
        'ETH',
        'LTC'
        ]

    models = [
        'FB_PROPHET',
        'SARIMAX',
        'LSTM'
        ]
    for curr in currency:
        for model in models:
            trainer = Trainer(curr)
            trainer.load_data()
            model_predict = {
                'FB_PROPHET': trainer.prophecy_predict,
                'SARIMAX': trainer.sarimax_prediction,
                'LSTM' : trainer.LSTM_predict
            }

            print('{} Model Prediction for {}'.format(model, curr))
            result = model_predict[model]()
            cache[model] = { curr : result }


@app.get("/")
def index():
    return cache

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
    if 'FB_PROPHET' in cache and selected_crypto in cache['FB_PROPHET']:
        return cache['FB_PROPHET'][selected_crypto]

    trainer = Trainer(selected_crypto)
    trainer.load_data()
    result = trainer.prophecy_predict()
    cache['FB_PROPHET'] = {selected_crypto : result}

    return cache['FB_PROPHET'][selected_crypto]

@app.get('/lstm_predict')
def predict_lstm(selected_crypto):

    pass

@app.get("/predict_model")
def predict_model(model, selected_crypto):
    '''
    Takes in two params model and crypto
    Returns Original data and Prediction in Json format
    '''
    if (model in cache) and (selected_crypto in cache[model]):
        return cache[model][selected_crypto]

    trainer = Trainer(selected_crypto)

    model_predict = {
        'FB_PROPHET': trainer.prophecy_predict,
        'SARIMAX': trainer.sarimax_prediction,
        'LSTM' : trainer.LSTM_predict
    }

    trainer.load_data()
    result = model_predict[model]()
    cache[model] = {selected_crypto: result}

    return cache[model][selected_crypto]
