from crypto_backend.data import get_data, organize_data, daily_data, get_LSTM_data_with_objective
from crypto_backend.transformers import LogTransformer
from crypto_backend.preprocessing import preprocessing_LSTM_data_and_get_generators
from crypto_backend.utils import init_and_compile_model, fit_LSTM_model, \
     LSTM_predict_with_generator, plot_LSTM_final_results
import crypto_backend.table as tables
from sklearn.pipeline import Pipeline, make_pipeline
from prophet import Prophet
import pmdarima as pm
import joblib
import pandas as pd
import tensorflow as tf
import numpy as np


class Trainer(object):
    def __init__(self, currency):
        '''
            X is a pandas dataframe, with the time column as the index
            y is the target for the forecast
        '''
        self.pipeFB = None
        self.pipeSARIMAX = None
        self.currency = currency
        self.X = None
        self.y = None
        # Default forecast objective is 'close'
        self.forecast_objective = 'close'

    #preprocessing pipeline
    def preproc_pipe_fb(self):
        self.pipeFB = Pipeline([
            ('LogTrans', LogTransformer()),
        ])

    def preproc_pipe_SA(self):
        self.pipeSARIMAX = Pipeline([
            ('LogTrans', LogTransformer())
        ])

    def load_data(self, daily_on = True):
        self.X = get_data(self.currency)
        self.X = organize_data(self.X)
        # if daily_on:
            # self.X = daily_data(self.X)
        self.lastday = self.X.index[-1]

    #generates a prophet model based on the coin
    def build_prophet(self):
        # Loading Data
        self.load_data()
        # initializing a prophet
        fbph = Prophet(seasonality_mode = 'multiplicative',
                        interval_width=0.95,
                        changepoint_prior_scale = 0.02,
                        yearly_seasonality = True,
                        weekly_seasonality = True,
                        daily_seasonality = True)
        # preprocess the data and changing it to the prophet format
        fbph.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        # prophet model performs better when it only sees the last year and half of data.
        data = self.X.iloc[-500:].copy()
        fb_data = data[['close']].reset_index()
        fb_data.rename(columns={'time':'ds','close':'y'},inplace = True)
        # prophet fitting and predicting
        fbph.fit(fb_data)
        #saving the model
        joblib.dump(fbph, f'models/{self.currency}_fb_prophet_model.joblib')

    #load saved Facebook Prophet Model and makes a 14-day prediction.
    def prophecy_predict(self,days=14):
        # fbph = joblib.load('prophet.joblib')
        #load the saved the model
        fbph = joblib.load(f'models/{self.currency}_fb_prophet_model.joblib')
        # making the prediction
        future= fbph.make_future_dataframe(periods=days,freq='d')
        forecast=fbph.predict(future)
        prediction = forecast.iloc[-14:]
        fb_data= {'data':self.X,'predict':prediction}
        return tables.make_fb_table(fb_data)


    def build_sarimax(self):
        #loading the Data
        self.load_data()
        #create pipeline
        self.preproc_pipe_SA()
        print('complete loading data and building pipeline')
        data = self.X.copy()
        data_t = self.pipeSARIMAX.fit_transform(data)
        model = pm.auto_arima(data_t['close'],
                            start_p=0, max_p=5,
                            start_q=0, max_q=5,
                            d=None,           # let model determine 'd'
                            test='adf',       # using adftest to find optimal 'd'
                            trace=True,
                            error_action='ignore',
                            suppress_warnings=True)
        joblib.dump(model, f'models/{self.currency}_sarimax_model.joblib')
        joblib.dump(self.pipeSARIMAX, f'models/{self.currency}_sarimax_model_pipe.joblib')

    #loads the sarimax model of the currency the make the 14 day prediction.
    def sarimax_prediction(self,days=14,return_conf_int=True):
        #loads the pre_made_sarimax model
        model = joblib.load(f'models/{self.currency}_sarimax_model.joblib')
        #loads the pipeline for the model
        self.pipeSARIMAX = joblib.load(f'models/{self.currency}_sarimax_model_pipe.joblib')
        #makes the n-day prediction
        forecast, conf_int = model.predict(days, return_conf_int = return_conf_int, alpha=0.05)
        #generate a n-day time range for the index of the results
        time_range = pd.date_range(start=self.lastday,periods =days+1)[1:]

        #insert the forecast data into a datafame that the transformer pipeline recognize
        d_temp = self.X.iloc[-1:].copy()
        d_temp = d_temp.append(pd.DataFrame({'close':forecast},index=time_range))
        d_temp.fillna(1)
        #inverse transform the data
        d_inv = self.pipeSARIMAX.inverse_transform(d_temp)
        d_inv =  d_inv.iloc[1:]
        conf_int = np.exp(conf_int)
        upper_end = pd.Series(conf_int[:,1],time_range)
        lower_end = pd.Series(conf_int[:,0],time_range)
        sarimax_data = {'data':self.X,'pred': d_inv['close'], 'upper':upper_end, 'lower':lower_end}
        results = tables.make_sarimax_table(sarimax_data)
        return results

    def build_LSTM(self, objective='close'):
        """Build and save the LSTM model with given forecast objective
        Return most of the params to be used in the forecast step"""
        self.forecast_objective = objective
        self.X, self.y = get_LSTM_data_with_objective(self.currency, self.forecast_objective)
        train_gen, val_gen, test_gen, index_70pct, index_85pct, scaler_X, scaler_y = preprocessing_LSTM_data_and_get_generators(self.X, self.y)
        model = init_and_compile_model(self.X)
        model = fit_LSTM_model(model, train_gen, val_gen)
        model.save(f'models/{self.currency}_LSTM_{self.forecast_objective}_model')
        return scaler_X, scaler_y, index_70pct, index_85pct, test_gen

    def LSTM_predict(self, objective='close'):
        """Get the prediction and plot final results with LSTM"""
        self.forecast_objective = objective
        self.X, self.y = get_LSTM_data_with_objective(self.currency, self.forecast_objective)
        train_gen, val_gen, test_gen, index_70pct, index_85pct, scaler_X, scaler_y = preprocessing_LSTM_data_and_get_generators(self.X, self.y)
        # scaler_X, scaler_y, index_70pct, index_85pct, test_gen = self.build_LSTM()
        model = tf.keras.models.load_model(f'models/{self.currency}_LSTM_{self.forecast_objective}_model')
        df_plot = LSTM_predict_with_generator(model, self.X, self.y, scaler_X, scaler_y, index_70pct, index_85pct, test_gen)
        # plot_LSTM_final_results(df_plot, self.currency)
        return df_plot

    def LSTM_multi_predict(self):
        # make 3 predictions for LSTM for the CI
        pred_close = self.LSTM_predict()['pred_future_price'].iloc[-14:]
        pred_high = self.LSTM_predict('high')['pred_future_price'].iloc[-14:]
        pred_low = self.LSTM_predict('low')['pred_future_price'].iloc[-14:]
        results = pd.concat([pred_close,pred_high,pred_low],axis=1)
        results = results.sort_values(by=results.index[0],axis=1).set_axis(['MIN Price',
                                                  'Predicted Price',
                                                  'MAX Price'],
                                                 axis = 'columns')
        results2= results.copy()
        for v,row in enumerate(results.iterrows()):
            results2.iloc[v]= np.sort(list(row)[1:])
        results = results.astype(float)
        results2 = results2.astype(float)
        # reload the date to get the X before the change by build_lstm
        self.load_data()
        lstm_data = {'data':self.X, 'predict':results2}
        # results = tables.make_LSTM_table(lstm_data)
        return lstm_data


if __name__ == '__main__':
    # Test function here
    # trainer = Trainer('BTC')
    # prediction = trainer.build_prophet()
    # prediction = trainer.prophecy_predict(days=1)
    # # print(trainer.X.iloc[0])
    # print(prediction['predict'])

    # Test SARIMAX & LSTM
    for coin in ['BTC']:
        trainer = Trainer(coin)
        # scaler_X, scaler_y, index_70pct, index_85pct, test_gen = trainer.build_LSTM()
        trainer.load_data()
        trainer.build_LSTM()
    # df_plot = trainer.LSTM_multi_predict()
    # print(df_plot)
    # train_gen, val_gen, test_gen, index_70pct, index_85pct, scaler_X, scaler_y = preprocessing_LSTM_data_and_get_generators(trainer.X, y)
    # model_api = init_and_compile_model()
    # model_api = fit_LSTM_model(model_api, train_gen, val_gen)
    # Save model to a folder
    # model_api.save(f'{current_path}/close_model')
    # df_api_plot = LSTM_predict_with_generator(model_api, X, y, scaler_X, scaler_y, index_70pct, index_85pct, test_gen)
    # plot_LSTM_final_results(df_api_plot, 'ETH')
