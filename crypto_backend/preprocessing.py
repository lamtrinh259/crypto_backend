from sklearn.preprocessing import MinMaxScaler
from crypto_backend.data import get_data

def MinMax_fit_transform(X):
    MinMaxscaler = MinMaxScaler()
    MinMaxscaler.fit(X)
    X_transformed = MinMaxscaler.fit_transform(X)
    return X_transformed, MinMaxscaler

def MinMax_reverse_fit_transform(MinMaxscaler, X_transformed):
    X = MinMaxscaler.inverse_transform(X_transformed)
    return X


if __name__ == '__main__':
    # For testing
    LTC = get_data('LTC')
    print(LTC)
    LTC_transformed, MinMaxscaler = MinMax_fit_transform(LTC)
    print(LTC_transformed)
    LTC_original = MinMax_reverse_fit_transform(LTC_transformed)
    print(LTC_original)
