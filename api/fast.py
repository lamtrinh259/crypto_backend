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
    currency = ['BTC', 'ETH', 'LTC']
    models = [
        'FB_PROPHET',
        'SARIMAX',
        # 'LSTM'
        ]
    for curr in currency:
        for model in models:
            trainer = Trainer(curr)
            trainer.load_data()
            model_build = {
                'FB_PROPHET': trainer.build_prophet,
                'SARIMAX': trainer.build_sarimax,
                'LSTM' : trainer.build_LSTM
            }
            model_predict = {
                'FB_PROPHET': trainer.prophecy_predict,
                'SARIMAX': trainer.sarimax_prediction,
                'LSTM' : trainer.LSTM_predict
            }
            if not isfile('{}_{}_model.joblib'.format(curr, model.lower())):
                print('Building {} for {}'.format(model, curr))
                model_build[model]()
            print('{} Model Prediction for {}'.format(model, curr))
            result = model_predict[model]()
            cache[model.lower()] = { curr : result }


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

@app.get("/predict_model")
def predict_model(model, selected_crypto):
    '''
    Takes in two params model and crypto
    Returns Original data and Prediction in Json format
    '''
    if model.lower() in cache and selected_crypto in cache[model.lower()]:
        return cache[model.lower()][selected_crypto]

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
