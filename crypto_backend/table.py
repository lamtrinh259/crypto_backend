import numpy as np
import pandas as pd
# from crypto_backend.trainer import Trainer
# uncomment above for local testing of the graphing functions


def make_fb_table(fb_data):
    # writes a table to be displayed on streamlit
    #grabbing the 7day data
    # prev_df = pd.DataFrame(fb_data['data'][['close','low','high']].iloc[-7:])
    # prev_df = prev_df.rename(columns={'close':'Predicted Price',
    #                                 'low': 'MIN Price',
    #                                 'high': 'MAX Price'})
    #grabbing the prediction data
    df = pd.DataFrame(fb_data['predict'][['ds','yhat','yhat_lower','yhat_upper']])
    fb_data['predict'] = df.rename(columns={'ds':'time',
                            'yhat':'Predicted Price',
                            'yhat_lower': 'MIN Price',
                            'yhat_upper': 'MAX Price'}).set_index('time')
    #merge the two table
    # results = pd.concat([prev_df,df])
    return fb_data

def make_sarimax_table(sarimax_data):
    # writes a table to be displayed on streamlit
    #grabbing the 7day data
    # prev_df = pd.DataFrame(sarimax_data['data'][['close','low','high']].iloc[-7:])
    # prev_df = prev_df.rename(columns={'close':'Predicted Price',
    #                                 'low': 'MIN Price',
    #                                 'high': 'MAX Price'})
    #grabbing the prediction data
    df = pd.DataFrame({'Predicted Price':sarimax_data['pred'],
                       'MIN Price':sarimax_data['lower'],
                       'MAX Price':sarimax_data['upper']})
    #merge the two table
    # results = pd.concat([prev_df,df])
    return df

def make_LSTM_table(lstm_data):
    # writes a table to be displayed on streamlit
    #grabbing the 7day data
    # prev_df = pd.DataFrame(lstm_data['data'][['close','low','high']].iloc[-7:])
    # prev_df = prev_df.rename(columns={'close':'Predicted Price',
    #                                 'low': 'MIN Price',
    #                                 'high': 'MAX Price'})
    #grabbing the prediction data
    df = lstm_data['pred']
    #merge the two table
    # results = pd.concat([prev_df,df])
    return df


if __name__ == '__main__':
    # Test function here

    trainer = Trainer('BTC')
    trainer.load_data()
    pred = trainer.LSTM_multi_predict()
    table = make_LSTM_table(pred)
    # table = make_fb_table(pred)
    print(table)

    pass
