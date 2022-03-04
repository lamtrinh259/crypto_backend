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


##Returns a dataframe with time in human-readable time as the index
def organize_data(df):
    df.time = pd.to_datetime(df.time, unit='ms')
    df_t = df.set_index('time')
    return df_t

##Returns a df with hourly data, requires a df where index is time.
def hourly_data(df,step=1):
    df_sampled = df[['open','close','high','low']].resample(f'{step}H').mean()
    df_sampled['volume'] = df[['volume']].resample(f'{step}H').sum()
    return df_sampled


##returns a df with daily data
def daily_data(df,step=1):
    df_sampled = df[['open','close','high','low']].resample(f'{step}D').mean()
    df_sampled['volume'] = df[['volume']].resample(f'{step}D').sum()
    return df_sampled

if __name__ == '__main__':
    # For testing
    # get_data('BTC')
    # get_data('ETH')
    get_data('LTC')
