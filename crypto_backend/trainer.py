from crypto_backend.data import get_data, organize_data
from crypto_backend.transformers import LogTransformer, DifferenceTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from prophet import Prophet
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

    def load_data(self):
        self.X = get_data(self.currency)
        self.X = organize_data(self.X)


    #Facebook Prophet Model that makes a 14-day prediction.
    def prophecy_predict(self,days=14):
        # Loading Data
        self.load_data()
        # Creating Pipeline
        self.preproc_pipe_fb()

        fbph = Prophet(seasonality_mode='multiplicative', interval_width=0.95 ,daily_seasonality=True)
        data = self.X.copy()
        fit_data = self.pipeFB.fit_transform(data)
        fit_data = fit_data[['close']].reset_index()
        fit_data.rename(columns={'time':'ds','close':'y'},inplace = True)
        fbph.fit(fit_data)
        future= fbph.make_future_dataframe(periods=days,freq='d')
        forecast=fbph.predict(future)
        forecast[['yhat','yhat_lower','yhat_upper','trend_lower','trend_upper']] = self.pipeFB.inverse_transform(
            forecast[['yhat','yhat_lower','yhat_upper','trend_lower','trend_upper']]
            )
        return {'data':self.X,'predict':forecast}

    #SARIMAX
    def sarimax_predict():
        pass


if __name__ == '__main__':
    # Test function here
    trainer = Trainer('BTC')
    prediction = trainer.prophecy_predict(days=1)
    print(trainer.X.iloc[0])
    print(prediction)
