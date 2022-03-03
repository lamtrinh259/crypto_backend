import pandas as pd

# Either URI or URL works
BTC_USD_S3URI = "s3://cryptocurrency-forecasting/raw_data/btcusd.csv"
BTC_USD_URL = "https://cryptocurrency-forecasting.s3.ap-northeast-1.amazonaws.com/raw_data/btcusd.csv"
ETH_USD_S3URI = "s3://cryptocurrency-forecasting/raw_data/etcusd.csv"
ETH_USD_URL = "https://cryptocurrency-forecasting.s3.ap-northeast-1.amazonaws.com/raw_data/etcusd.csv"
LTC_USD_S3URI = "s3://cryptocurrency-forecasting/raw_data/ltcusd.csv"
LTC_USD_URL = "https://cryptocurrency-forecasting.s3.ap-northeast-1.amazonaws.com/raw_data/ltcusd.csv"

def get_data(crypto, last_rows=500000):
    """ Get dataset of selected crypto from cloud storage
    500,000 data points (mins) is roughly equivalent to 1 year worth of data
    # Params: chosen crypto by the user from the front end """
    if crypto == "BTC":
        df_full = pd.read_csv(BTC_USD_URL)
        n_rows = len(df_full)
        df = pd.read_csv(BTC_USD_URL, skiprows=range(1, n_rows - last_rows))
    elif crypto == 'ETH':
        df_full = pd.read_csv(ETH_USD_URL)
        n_rows = len(df_full)
        df = pd.read_csv(ETH_USD_URL, skiprows=range(1, n_rows - last_rows))
    elif crypto == 'LTC':
        df_full = pd.read_csv(LTC_USD_S3URI)
        n_rows = len(df_full)
        df = pd.read_csv(LTC_USD_S3URI, skiprows=range(1, n_rows - last_rows))
    return df

BTC_local_path = 'raw_data/btcusd.csv'
ETH_local_path = 'raw_data/ethusd.csv'
LTC_local_path = 'raw_data/ltcusd.csv'

def get_data_locally(crypto, last_rows=500000):
    """ Get dataset of selected crypto from cloud storage
    500,000 data points (mins) is roughly equivalent to 1 year worth of data
    # Params: chosen crypto by the user from the front end """
    df = None
    if crypto == 'BTC':
        df_full = pd.read_csv(BTC_local_path)
        n_rows = len(df_full)
        df = pd.read_csv(BTC_local_path, skiprows=range(1, n_rows - last_rows))
    elif crypto == 'ETH':
        df_full = pd.read_csv(ETH_local_path)
        n_rows = len(df_full)
        df = pd.read_csv(ETH_local_path, skiprows=range(1, n_rows - last_rows))
    elif crypto == 'LTC':
        df_full = pd.read_csv(LTC_local_path)
        n_rows = len(df_full)
        df = pd.read_csv(LTC_local_path, skiprows=range(1, n_rows - last_rows))
    return df

##Returns a dataframe with time in human-readable time as the index
def organize_data(df):
    df.time = pd.to_datetime(df.time, unit='ms')
    df_t = df.set_index('time')
    return df_t

def get_X_y(df):
    y = df['close']
    X = df.drop('close', axis=1)
    return X, y

if __name__ == '__main__':
    # For testing
    BTC = get_data_locally('BTC')
    print(BTC)
    # get_data('BTC')
    # get_data('ETH')
    # get_data('LTC')
