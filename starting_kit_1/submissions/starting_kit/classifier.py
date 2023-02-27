import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler):
        self.scaler = scaler

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X_df):
        idx = X_df.index
        col = X_df.columns
        res = self.scaler.transform(X_df)
        res = pd.DataFrame(res,index=idx,columns=col)

        return res
    
def maxvariance(X: pd.DataFrame, threshold: int = 20) -> pd.DataFrame: 

    """
    For a given dataframe, keeps the number features with the highest variance

    Parameters
    ----------
    X : dataframe
    threshold : int
        the number of features with the highest variance
    """
    var = X.var()
    var = var.sort_values()
    selected = var.tail(threshold)
    col = list(selected.index.values)
    X = X.loc[:, col]

    return X

class FeatureSelector(BaseEstimator):
    def fit (self, X, y):
        return self
    
    def transform(self, X, y):
        return maxvariance(X, threshold = 20)
  

class Classifier(BaseEstimator):
    def __init__(self):
        self.scaling = Scaler(StandardScaler())
        self.logreg = LogisticRegression(max_iter=100)
        self.cls = make_pipeline(self.scaling, self.logreg)
    
    def fit(self, X, y):
        self.cls.fit(X, y)
    
    def predict(self, X):
        return self.clf.predict(X)
    
    def predict_proba(self, X):
        return self.cls.predict_proba(X)
  

def get_estimator():

    featureselector = FeatureSelector()
    scaling = Scaler(StandardScaler())
    logreg = LogisticRegression(max_iter=100)

    pipe = make_pipeline(scaling, featureselector, logreg)

    return pipe