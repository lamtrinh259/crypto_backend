import numpy as np
import pandas as pd

def upper_shadow(df): return df['high'] - np.maximum(df['close'], df['open'])
def lower_shadow(df): return np.minimum(df['close'], df['open']) - df['low']

def get_features(df, row = False):
    """ Transform the df into a df with basic features and dropna"""
    df_feat = df.copy()
    df_feat['spread'] = df_feat['high'] - df_feat['low']
    df_feat['upper_shadow'] = upper_shadow(df_feat)
    df_feat['lower_shadow'] = lower_shadow(df_feat)
    df_feat['close-open'] = df_feat['close'] - df_feat['open']
    df_feat['SMA_7'] = df_feat.iloc[:,1].rolling(window=7).mean()
    df_feat['SMA_14'] = df_feat.iloc[:,1].rolling(window=14).mean()
    df_feat['SMA_21'] = df_feat.iloc[:,1].rolling(window=21).mean()
    # Create the STD_DEV feature for the past 7 days
    df_feat['STD_DEV_7'] = df_feat.iloc[:,1].rolling(window=7).std()
    # Donchian Channel using pandas rolling min/max instead of pandas_ta
    df_feat['DCL_10_15'] = df_feat['low'].rolling(window=10).min()
    df_feat['DCU_15'] = df_feat['high'].rolling(window=15).max()
    df_feat['DCM_10_15'] = (df_feat['DCL_10_15'] + df_feat['DCU_15']) / 2
    # Drop the NA rows created by the SMA indicators
    df_feat.dropna(inplace = True)
    return df_feat

if __name__ == '__main__':
    pass
    # For testing, it works!
    # ETH = get_data('ETH', last_rows=1000)
    # ETH = get_features(ETH)
    # print(ETH)
