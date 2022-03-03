from crypto_backend.data import get_data, organize_data
from crypto_backend.transformers import LogTransformer, DifferenceTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from prophet import Prophet

class Trainer(object):
    def __init__(self, X):
        '''
            X is a pandas dataframe, with the time column as the index
        '''
        self.pipeA = None
        self.X = X


    #preprocessing pipeline
    def preproc_pipe(self):
        self.pipeA = Pipeline([
            ('LogTrans', LogTransformer()),
        ])

    #Facebook Prophet Model that makes a 14-day prediction.
    def prophecy_predict(self,days=14):
        fbph = Prophet(seasonality_mode='multiplicative', interval_width=0.95 ,daily_seasonality=True)
        data = self.X.copy()
        fit_data = self.pipeA.fit_transform(data)
        fit_data = fit_data[['close']].reset_index()
        fit_data.rename(columns={'time':'ds','close':'y'},inplace = True)
        fbph.fit(fit_data)
        future= fbph.make_future_dataframe(periods=days,freq='d')
        forecast1=fbph.predict(future)
        forecast1[['trend','yhat_lower','yhat_upper','trend_lower','trend_upper']] = self.pipeA.inverse_transform(
            forecast1[['trend','yhat_lower','yhat_upper','trend_lower','trend_upper']]
            )
        return forecast1

    #SARIMAX
    def sarimax_predict():
        pass


if __name__ == '__main__':
    # Test function here
    data = get_data('BTC')
    clean_data = organize_data(data)
    print(clean_data.iloc[0])
    trainer = Trainer(clean_data)
    trainer.preproc_pipe()
    prediction = trainer.prophecy_predict(days=1)
    print(prediction)
