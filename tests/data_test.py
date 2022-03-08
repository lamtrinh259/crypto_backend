import crypto_backend.data as data_man
import unittest
import datetime
import pandas as pd
class DataTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.currency = 'BTC'
        cls.data = data_man.get_data(cls.currency)
        cls.organized_data = data_man.organize_data(cls.data)
        cls.daily_data = data_man.daily_data(cls.organized_data)
        cls.hourly_data = data_man.hourly_data(cls.organized_data)
        # cls.X,cls.y = data_man.get_X_y(cls.daily_data)
        cls.X, cls.y = data_man.get_LSTM_data_with_objective(cls.currency, 'close')

    # def setUp(self):
    #     #this will run before every test
    #     self.currency = 'BTC'
    #     self.data = data_man.get_data(self.currency)


    def test_getdatatype(cls):
        #test get_data returns dataframe
        cls.assertEqual(type(cls.data),pd.DataFrame)
        # assert type(cls.data) == pd.DataFrame

    def test_getdatacolumns(cls):
        #test get_data returns the correct columns
        cols = ['time','open','close','high','low','volume'].sort()
        cls.assertEqual(list(cls.data.columns).sort(),cols)

    def test_organize(cls):
        #test organize data returns a df with time as index
        cls.assertEqual(type(cls.organized_data.index[-1]),
                             pd._libs.tslibs.timestamps.Timestamp)

    def test_daily(cls):
        #test the time difference between the last two rows to make sure its daily
        cls.assertEqual(cls.daily_data.index[-1]-cls.daily_data.index[-2],
                        datetime.timedelta(days=1))

    def test_daily_nonan(cls):
        #check if there are any nan values in the data
        cls.assertEqual(cls.daily_data.isna().sum().sum(),0)

    def test_hourly(cls):
        #test the time difference of the last two rows is an hour
        cls.assertEqual(cls.hourly_data.index[-1]-cls.hourly_data.index[-2],
                        datetime.timedelta(hours=1))

    def test_hourly_nonan(cls):
        #check if there are any nan values in the data
        cls.assertEqual(cls.hourly_data.isna().sum().sum(),0)

    # def test_X_y(cls):
    #
    def test_x_y(cls):
        cls.assertEqual(type(cls.y),pd.Series)
        cols = ['open','high','low','volume',"close"].sort()
        cls.assertEqual(list(cls.X.columns).sort(),
                        cols)
        cls.assertEqual(len(cls.X),len(cls.y))

    def test_getapidata_shape(cls):
        data = data_man.get_data_from_api(frames=100)
        cls.assertEqual(data.shape, (100,6))



if __name__ == '__main__':
    tester = DataTester()
    tester.test_getdatacolumns()
    tester.test_getdatatype()
