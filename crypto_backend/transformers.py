'''
Transformer classes for the pipeline
'''
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np

#Log transforms the data set
class LogTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        X_linearized = np.log(X)
        return X_linearized
        if y:
            y_l = np.log(y)
            return y_l

    def inverse_transform(self, X, y=None):
        return np.exp(X)
        if y:
            y_ex = np.exp(y)
            return y_ex


#returns the running diff set of a data set
class DifferenceTransformer(TransformerMixin,BaseEstimator):
    def __init__(self):
        pass

    def fit(self,X,y=None):
        self.init_term = X[0]
        self.final_term = X[-1]
        return self

    def transform(self,X,y=None):
        X_diff = X[:-1]-X[1:]
        return X_diff

    def inverse_transform(self, X, y=None):
        X_inversed = []
        i = self.final_term
        for row in X():
            i += row
            X_inversed.append(i)
        return X_inversed
