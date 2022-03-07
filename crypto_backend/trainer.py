from crypto_backend.data import get_data, organize_data, daily_data
from crypto_backend.transformers import LogTransformer, DifferenceTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from prophet import Prophet
import pmdarima as pm
import joblib
import pandas as pd


class Trainer(object):
    def __init__(self, currency):
        '''
            X is a pandas dataframe, with the time column as the index
        '''
        self.pipeFB = None
        self.pipeSARIMAX = None
        self.currency = currency
        self.X = None

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
        if daily_on:
            self.X = daily_data(self.X)
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
        joblib.dump(fbph, f'{self.currency}_prophet_model.joblib')



    #load saved Facebook Prophet Model and makes a 14-day prediction.
    def prophecy_predict(self,days=14):
        # fbph = joblib.load('prophet.joblib')
        #load the saved the model
        fbph = joblib.load(f'{self.currency}_prophet_model.joblib')
        # making the prediction
        future= fbph.make_future_dataframe(periods=days,freq='d')
        forecast=fbph.predict(future)

        return {'data':self.X,'predict':forecast}

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
        joblib.dump(model, f'{self.currency}_sarimax_model.joblib')
        joblib.dump(self.pipeSARIMAX, f'{self.currency}_sarimax_model_pipe.joblib')

    #loads the sarimax model of the currency the make the 14 day prediction.
    def sarimax_prediction(self,days=14,return_conf_int=True):
        #loads the pre_made_sarimax model
        model = joblib.load(f'{self.currency}_sarimax_model.joblib')
        #loads the pipeline for the model
        self.pipeSARIMAX = joblib.load(f'{self.currency}_sarimax_model_pipe.joblib')
        #makes the n-day prediction
        forecast, conf_int = model.predict(days, return_conf_int = return_conf_int, alpha=0.05)
        #generate a n-day time range for the index of the results
        time_range = pd.date_range(start=self.lastday,periods =days+1)[1:]

        #insert the forcast data into a datafame that the transformer pipeline recognize
        d_temp = self.X.iloc[-1:].copy()
        d_temp = d_temp.append(pd.DataFrame({'close':forecast},index=time_range))
        d_temp.fillna(1)
        #inverse transform the data
        d_inv = self.pipeSARIMAX.inverse_transform(d_temp)
        upper_end = pd.Series(conf_int[:,0],time_range)
        lower_end = pd.Series(conf_int[:,1],time_range)
        return d_inv['close'], upper_end, lower_end


if __name__ == '__main__':
    # Test function here
    trainer = Trainer('BTC')
    prediction = trainer.build_prophet()
    prediction = trainer.prophecy_predict(days=1)
    # print(trainer.X.iloc[0])
    print(prediction['predict'])
