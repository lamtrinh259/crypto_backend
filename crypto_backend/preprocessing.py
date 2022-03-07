from sklearn.preprocessing import MinMaxScaler
from crypto_backend.data import get_data
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator

def preprocessing_LSTM_data_and_get_generators(X, y):
    """ Preprocessing data with MinMax scaler and return the 3 generators, the 2 index percentiles
    for val and test sets, and the respective X & y scalers to be used with prediction.
    Params: X (features dataset) and y (target) """
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = MinMaxScaler()
    values_y = y.values.reshape(-1, 1)
    y_scaled = scaler_y.fit_transform(values_y)
    print('Shape of data after scaled is', X_scaled.shape, y_scaled.shape)

    # Set the index to divide data into train, val, and test sets
    index_70pct = int(len(X)*0.7) # end of training set
    index_85pct = int(len(X)*0.85) # end of val set and beginning of test set
    print('The ending index for training set is', index_70pct)
    print('The beginning index for test set (also the ending index for val set) is', index_85pct)

    # Define time series generator for training, validation, and test sets
    seq_len = 30 # Length of each sequence
    batch_size = 32 # Number of observations
    n_features = X_scaled.shape[1]

    train_generator = TimeseriesGenerator(X_scaled, y_scaled, length=seq_len, batch_size=batch_size, end_index = index_70pct)
    val_generator = TimeseriesGenerator(X_scaled, y_scaled, length=seq_len, batch_size=batch_size, \
                                        start_index = index_70pct, end_index = index_85pct)
    test_generator = TimeseriesGenerator(X_scaled, y_scaled, length=seq_len, batch_size=batch_size, start_index = index_85pct)
    return train_generator, val_generator, test_generator, index_70pct, index_85pct, scaler_X, scaler_y


if __name__ == '__main__':
    # For testing
    LTC = get_data('LTC')
