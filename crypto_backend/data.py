# Pandas error
import pandas as pd

BTC_USD_S3URI = "s3://cryptocurrency-forecasting/raw_data/btcusd.csv"
BTC_USD_URL = "https://cryptocurrency-forecasting.s3.ap-northeast-1.amazonaws.com/raw_data/btcusd.csv"
ETH_USD_S3URI = "s3://cryptocurrency-forecasting/raw_data/etcusd.csv"
ETH_USD_URL = "https://cryptocurrency-forecasting.s3.ap-northeast-1.amazonaws.com/raw_data/etcusd.csv"
LTC_USD_S3URI = "s3://cryptocurrency-forecasting/raw_data/ltcusd.csv"
LTC_USD_URL = "https://cryptocurrency-forecasting.s3.ap-northeast-1.amazonaws.com/raw_data/ltcusd.csv"

def get_data(crypto, last_rows=5000):
    """ Get dataset of selected crypto from cloud storage"""
    # if crypto == "BTC":
    # df = pd.read_csv(f'{crypto}_USD_path', nrows)
    fname = f'{crypto}_USD_URL'
    n_rows = sum(1 for row in open(fname, 'r'))
    df = pd.read_csv(fname, skiprows=range(1, n_rows - last_rows))
    print(df.head())
    return df



if __name__ == '__main__':
    # Test function here
    # For testing
    get_data('BTC')
    pass
