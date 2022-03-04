'''
Tools to be used during the operations
'''
from cgi import test
import plotly.graph_objects as go
import crypto_backend.data as datar

def time_series_generator(X, y):
    pass


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
    data = datar.get_data(currency)
    odata = datar.organize_data(data)
    d_data = datar.daily_data(odata)
    fig1 = go.Figure(data=[go.Candlestick(x=d_data['time'],
                                        open=d_data['open'],
                                        high=d_data['high'],
                                        low=d_data['low'],
                                        close=d_data['close']),
                go.Scatter(x=fb_data.ds.iloc[-14:],
                            y=fb_data.yhat.iloc[-14:],
                            mode='lines')
                ])
    return fig1


if __name__ == '__main__':
    # Test function here
    pass
