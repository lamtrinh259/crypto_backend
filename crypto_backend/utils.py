# Tools to be used during the operations
from cgi import test
import plotly.graph_objects as go
import crypto_backend.data as datar
import pandas as pd
import matplotlib.pyplot as plt
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import numpy as np
from datetime import timedelta

def init_and_compile_model(X):
    """ Initialize and compile the LSTM model with Adam optimizer"""
    seq_len = 30
    n_features = X.shape[1]
    model = Sequential()
    model.add(LSTM(name='lstm_1st_layer', units=128, return_sequences=True, activation='relu', input_shape=(seq_len, n_features)))
    model.add(LSTM(name='lstm_2nd_layer', units=64, activation='tanh'))
    model.add(Dense(units=64, name='dense_1st_layer', activation = 'LeakyReLU'))
    model.add(Dropout(0.2, name='dropout_layer',))
    model.add(Dense(1, name='final_layer', activation='linear'))
    Adam_opt = tf.keras.optimizers.Adam(0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='mse', optimizer = Adam_opt, metrics='mse')
    return model

def fit_LSTM_model(model, train_generator, val_generator):
    """ Fit an LSTM model with training data and validate on validation data"""
    current_path = os.getcwd() # current directory
    # Run and fit the model
    es = EarlyStopping(patience=10, monitor='val_loss')
    cp = ModelCheckpoint(f'{current_path}/checkpoints', monitor='val_loss', save_best_only=True)
    history = model.fit(train_generator, validation_data = val_generator, epochs=50, verbose=1, callbacks=[es, cp])
    # plt.plot(history.history['loss'], label='training data')
    # plt.plot(history.history['val_loss'], label='validation data')
    # plt.legend()
    # plt.show()
    return model

def LSTM_predict_with_generator(model, X, y, scaler_X, scaler_y, index_70pct, index_85pct, test_generator):
    """ Pass in the model, the original X and y data, along with the 70%, 85% and the test generator in order to
    predict the future 14 days with the LSTM model and return the entire historical and predicted data as 1 df"""
    # Actual prices from the entire data
    df_actual = pd.DataFrame({'actual_price':  y.values, 'date': y.index})

    # Predicted prices from the given generator set
    y_in_sample_pred = model.predict(test_generator)
    y_in_sample_pred_prices = scaler_y.inverse_transform(y_in_sample_pred)
    days_in_sample_pred = y[index_85pct+30:].index
    # print(days_in_sample_pred)
    # print(len(BTC_y_pred_prices), len(days_in_sample_pred))
    df_pred_in_sample = pd.DataFrame({'pred_in_sample_price': y_in_sample_pred_prices.reshape(len(y_in_sample_pred_prices)), \
                                    'date': days_in_sample_pred})

    # Get the last 30 days of X, and it'll predict the forecast objective of 14 days later
    scaled_X = scaler_X.transform(X)
    seq_X = []
    for i in range(14):
        seq_X.append(scaled_X[-44+i:-14+i])
    seq_X = np.array(seq_X)
    # Predict the future 14 days from the last 44-day values
    y_pred = model.predict(seq_X)
    y_pred_prices = scaler_y.inverse_transform(y_pred)
    # Get the values of the extra days for future predicted prices
    days_future_pred = y[-14:].index + timedelta(14)
    y_pred_df = pd.DataFrame({'pred_future_price': y_pred_prices.reshape(14,), 'date': days_future_pred})
    y_pred_df.set_index('date', inplace=True)

    df_plot = df_actual.merge(df_pred_in_sample, how='outer', on='date').merge(y_pred_df, how='outer', on='date')
    df_plot['pred_future_price'].fillna(df_plot['pred_in_sample_price'], inplace=True)
    # print('Number of missing data points in each column is', df_plot.isnull().sum())
    df_plot.set_index('date', inplace=True)
    return df_plot

def plot_LSTM_final_results(df_plot, crypto):
    """ Plot the final results with actual prices and predicted prices (from test generator) for given crypto """
    plt.plot(df_plot['actual_price'], color = 'g', label = f'Actual prices of {crypto}')
    plt.plot(df_plot['pred_future_price'], color = 'b', label = f'Predicted prices of {crypto}')
    plt.legend(loc='best')
    plt.xlabel('Date')
    plt.ylabel('Price in USD')
    plt.title(f'Actual and predicted prices of {crypto} in USD')
    plt.show()

# Memory saving function that can only be used with df
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype.name

        if col_type not in ['object', 'category', 'datetime64[ns, UTC]', 'datetime64[ns]']:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# This train test generator's default assumes the following,
# approximately 2 years of data is input into the function
# It then makes 20 4-month-long train_test_split sets where the first 3 month is train and last month is test
# each set is 1 month further than the previous set . this can be changed by changing the step

def train_test_generator(X,test_sets=20,train_size=3*30*24*60,test_size=30*24*60,step = 30*24*60):
    X_train= []
    X_test=[]
    for i in range(test_sets):
        #start at the beginning of the set
        start= i*(step)
        #training set
        X_train.append(X.iloc[(start):(start+train_size)])
        #testing set
        X_test.append(X.iloc[(start+train_size):(start+train_size+test_size)])

    return X_train,X_test

#take in prediction take from facebook then graphs it together with previous data
def FB_grapher(fb_data,currency):
    # data = datar.get_data(currency)
    # odata = datar.organize_data(data)
    # d_data = datar.daily_data(odata)
    d_data = fb_data['data'].reset_index()
    pred_x = fb_data['pred'].ds.iloc[-14:]

    fig1 = go.Figure(data=[
                go.Candlestick(x=d_data['time'],
                            open=d_data['open'],
                            high=d_data['high'],
                            low=d_data['low'],
                            close=d_data['close'],
                            name = 'Historical Data'),
                go.Scatter(x=pred_x,
                            y=fb_data['pred'].yhat.iloc[-14:],
                            mode='lines',
                            name = 'Prediction'),
                go.Scatter( x=pred_x , # x, then x reversed
                            y=fb_data['pred'].yhat_upper[-14:]+fb_data['pred'].yhat_lower[-14::-1], # upper, then lower reversed
                            fill='toself',
                            fillcolor='rgba(0,100,80,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo="skip",
                            showlegend=False)
                ])
    return fig1


#take in prediction in the form returned fom sarimax_predict then graphs it together with previous data

def sarimax_grapher(sarimax_data,currency):
    # data = datar.get_data(currency)
    # odata = datar.organize_data(data)
    # d_data = datar.daily_data(odata)
    d_data = sarimax_data['data'].reset_index()
    pred_x = sarimax_data.pred.index
    fig2 = go.Figure(data=[
                go.Candlestick(x=d_data['time'],
                            open=d_data['open'],
                            high=d_data['high'],
                            low=d_data['low'],
                            close=d_data['close'],
                            name = 'Historical Data'),
                go.Scatter( x=pred_x,
                            y=sarimax_data.pred.close,
                            mode='lines',
                            name = 'Prediction'),
                go.Scatter( x=pred_x +pred_x[::-1], # x, then x reversed
                            y=sarimax_data.upper+sarimax_data.lower[::-1], # upper, then lower reversed
                            fill='toself',
                            fillcolor='rgba(0,100,80,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo="skip",
                            showlegend=False)
                ])
    return fig2

def all_grapher(input_data,currency):
    # data = datar.get_data(currency)
    # odata = datar.organize_data(data)
    # d_data = datar.daily_data(odata)
    d_data = input_data.data
    pred_x = input_data.pred.index[-14:]
    end_date = input_data.pred.index[-1]
    start_date = end_date-timedelta(days=30)

    fig2 = go.Figure(data=[
                go.Candlestick(x = d_data.index,
                            open = d_data['open'],
                            high = d_data['high'],
                            low = d_data['low'],
                            close = d_data['close'],
                            name = 'Historical Data'),
                go.Scatter( x = pred_x,
                            y = input_data.pred['Predicted Price'],
                            mode = 'lines',
                            name = 'Prediction'),
                go.Scatter( x=pred_x, # +pred_x[::-1], # x, then x reversed,
                            y=input_data.pred['MAX Price'] , # upper, then lower reversed
                            mode = 'lines',
                            line=dict(width=0),
                            hoverinfo="skip",
                            showlegend=False),
                go.Scatter( x=pred_x, # +pred_x[::-1], # x, then x reversed,
                            y=input_data.pred['MIN Price'], # upper, then lower reversed
                            fill='tonexty',
                            mode = 'lines',
                            line=dict(width=0),
                            fillcolor='rgba(68, 68, 68, 0.3)',
                            hoverinfo="skip",
                            showlegend=False)])
    fig2.update_xaxes(type="date", range=[start_date, end_date])
    return fig2


if __name__ == '__main__':
    # Test function here
    pass
