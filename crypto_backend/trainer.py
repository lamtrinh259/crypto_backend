from crypto_backend.data import get_data, organize_data, daily_data
from crypto_backend.transformers import LogTransformer, DifferenceTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from prophet import Prophet
# import pmdarima as pm
import joblib


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
            ('LogTrans', LogTransformer()),
            ('diff', DifferenceTransformer())
        ])

    def load_data(self, daily_on = True):
        self.X = get_data(self.currency)
        self.X = organize_data(self.X)
        if daily_on:
            self.X = daily_data(self.X)


    #Facebook Prophet Model that makes a 14-day prediction.
    def prophecy_predict(self,days=14):
        # Loading Data
        # self.load_data()
        # Creating Pipeline
        # self.preproc_pipe_fb()
        # # initializing a prophet
        # fbph = Prophet(seasonality_mode='multiplicative', interval_width=0.95 ,daily_seasonality=True)
        # # preprocess the data and changing it to the prophet format
        # data = self.X.copy()
        # # fit_data = self.pipeFB.fit_transform(data)
        # # fit_data = fit_data[['close']].reset_index()
        # fit_data = data[['close']].reset_index()
        # fit_data.rename(columns={'time':'ds','close':'y'},inplace = True)

        # # prophet fitting and predicting
        # fbph.fit(fit_data)
        fbph = joblib.load('prophet.joblib')
        future= fbph.make_future_dataframe(periods=days,freq='d')
        forecast=fbph.predict(future)

        # # inverse transforming the data back to numbers
        # forecast[['yhat','yhat_lower','yhat_upper','trend_lower','trend_upper']] = self.pipeFB.inverse_transform(
        #     forecast[['yhat','yhat_lower','yhat_upper','trend_lower','trend_upper']]
        #     )
        return {'data':self.X,'predict':forecast}

    #SARIMAX
    def sarimax_predict(self,days=14):
        #loading the Data
        self.load_data()
        #create pipeline
        self.preproc_pipe_SA()

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

        #model assumes the input data is a daily df so it makes a 14 day prediction
        model.predict(days)



if __name__ == '__main__':
    # Test function here
    trainer = Trainer('BTC')
    prediction = trainer.prophecy_predict(days=1)
    # print(trainer.X.iloc[0])
    print(prediction)
