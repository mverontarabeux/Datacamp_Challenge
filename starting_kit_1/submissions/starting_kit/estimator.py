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
  
def get_estimator():
    scaling = Scaler(StandardScaler())
    logreg = LogisticRegression(max_iter=100)

    pipe = make_pipeline(scaling, logreg)

    return pipe