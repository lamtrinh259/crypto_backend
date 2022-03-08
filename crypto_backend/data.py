import pandas as pd
from crypto_backend.feature_engineering import get_features
from datetime import datetime
import urllib.parse
import requests

# Either URI or URL works
# Complete minute data
# BTC_USD_S3URI = "s3://cryptocurrency-forecasting/raw_data/btcusd.csv"
# BTC_USD_URL = "https://cryptocurrency-forecasting.s3.ap-northeast-1.amazonaws.com/raw_data/btcusd.csv"
# ETH_USD_S3URI = "s3://cryptocurrency-forecasting/raw_data/ethusd.csv"
# ETH_USD_URL = "https://cryptocurrency-forecasting.s3.ap-northeast-1.amazonaws.com/raw_data/ethusd.csv"
# LTC_USD_S3URI = "s3://cryptocurrency-forecasting/raw_data/ltcusd.csv"
# LTC_USD_URL = "https://cryptocurrency-forecasting.s3.ap-northeast-1.amazonaws.com/raw_data/ltcusd.csv"

# Daily data from Jan 1st 2020 to Mar 7th 2022
BTC_USD_S3URI = "s3://cryptocurrency-forecasting/raw_data/BTCUSD_1D_2022-03-07.csv"
BTC_USD_URL = "https://cryptocurrency-forecasting.s3.ap-northeast-1.amazonaws.com/raw_data/BTCUSD_1D_2022-03-07.csv"
ETH_USD_S3URI = "s3://cryptocurrency-forecasting/raw_data/ETHUSD_1D_2022-03-07.csv"
ETH_USD_URL = "https://cryptocurrency-forecasting.s3.ap-northeast-1.amazonaws.com/raw_data/ETHUSD_1D_2022-03-07.csv"
LTC_USD_S3URI = "s3://cryptocurrency-forecasting/raw_data/LTCUSD_1D_2022-03-07.csv"
LTC_USD_URL = "https://cryptocurrency-forecasting.s3.ap-northeast-1.amazonaws.com/raw_data/LTCUSD_1D_2022-03-07.csv"

def get_data(crypto):
    """ Get daily dataset of selected crypto from cloud storage
    # Params: chosen crypto by the user from the front end """
    df = None
    if crypto == "BTC":
        df = pd.read_csv(BTC_USD_URL)
    elif crypto == 'ETH':
        df = pd.read_csv(ETH_USD_URL)
    elif crypto == 'LTC':
        df = pd.read_csv(LTC_USD_S3URI)
    else:
        print('No such crypto pair or file exists, please check')
    return df

BTC_local_path = 'raw_data/BTCUSD_1D_2022-03-07.csv'
ETH_local_path = 'raw_data/ETHUSD_1D_2022-03-07.csv'
LTC_local_path = 'raw_data/LTCUSD_1D_2022-03-07.csv'

def get_data_locally(crypto):
    """ Get dataset of selected crypto from local drive
    # Params: chosen crypto by the user from the front end """
    df = None
    if crypto == 'BTC':
        df = pd.read_csv(BTC_local_path)
    elif crypto == 'ETH':
        df = pd.read_csv(ETH_local_path)
    elif crypto == 'LTC':
        df = pd.read_csv(LTC_local_path)
    else:
        print('No such crypto pair or file exists, please check')
    return df

def get_data_from_api(time='1m',currency='BTCUSD',section='hist',frames=10_000,save_locally = True):
    BASE_URI = "https://api-pub.bitfinex.com"
    '''
    getting the data for the currency in the time difference for the number of frames,
    max number of frames is 10_000
    default setting is returning the last 10000 minutes for the btcusd set
    '''
    search_url = urllib.parse.urljoin(BASE_URI,f'/v2/candles/trade:{time}:t{currency}/{section}?limit={frames}')
    response = requests.get(search_url).json()
    if len(response)==frames:
        df = pd.DataFrame(response[::-1]).set_axis(['time','open','close','high','low','volume'],axis='columns')
        if save_locally:
            df.to_csv(f'local_data/{currency}_{time}_{str(datetime.now().date())}.csv')
        return df
    return None


def organize_data(df):
    """Returns a dataframe with time in human-readable time as the index"""
    df.time = pd.to_datetime(df.time, unit='ms')
    df_t = df.set_index('time')
    return df_t


def hourly_data(df,step=1):
    """Returns a df with hourly data, requires a df where index is time.
    Params: df with time set as index"""
    df_sampled = df[['open','close','high','low']].resample(f'{step}H').mean()
    df_sampled = df[['open']].resample(f'{step}H').first()
    df_sampled['close'] = df[['close']].resample(f'{step}H').last()
    df_sampled['high'] = df[[ 'high' ]].resample(f'{step}H').max()
    df_sampled['low']= df[[ 'low' ]].resample(f'{step}H').min()
    df_sampled['volume'] = df[['volume']].resample(f'{step}H').sum()
    df_sampled.interpolate(inplace=True)
    return df_sampled



def daily_data(df,step=1):
    """Returns a df with daily data
    Params: df with time set as index"""
    df_sampled = df[['open']].resample(f'{step}D').first()
    df_sampled['close'] = df[['close']].resample(f'{step}D').last()
    df_sampled['high'] = df[[ 'high' ]].resample(f'{step}D').max()
    df_sampled['low']= df[[ 'low' ]].resample(f'{step}D').min()
    df_sampled['volume'] = df[['volume']].resample(f'{step}D').sum()
    df_sampled.interpolate(inplace=True)
    return df_sampled

def get_LSTM_data_with_objective(crypto, forecast_objective):
    """ User will pass in the crypto name and the corresponding objective: 'close', 'high', 'low', 'open', and the dataset
    for use with LSTM model will be generated
    Returns: X and y"""
    df = get_data(crypto)
    df = organize_data(df)
    # Slice data if we happen to have data older than 2020 so that we only use data from the beginning of 2020
    cutoff_date = datetime.strptime('2020-01-01', '%Y-%M-%d')
    if df.index[0] < cutoff_date:
        df = df[cutoff_date:]
    df = get_features(df)
    y = df[forecast_objective][14:] # Target, exclude the first 14 days
    X = df[:-14]  # Exclude the last 14 days, all columns will be used as features
    return X, y

if __name__ == '__main__':
    # For testing
    # get_data_from_api(time='1D',frames=797)
    # get_data_from_api(time='1D',currency = 'ETHUSD',frames=797)
    # get_data_from_api(time='1D',currency = 'LTCUSD',frames=797)
    get_data('BTC')
    # get_data('ETH')
    # get_data('LTC')
