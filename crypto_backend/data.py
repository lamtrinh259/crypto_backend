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
    print(df.head())
    return df



if __name__ == '__main__':
    # For testing
    # get_data('BTC')
    # get_data('ETH')
    get_data('LTC')
