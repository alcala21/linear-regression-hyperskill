# write your code here
import pandas as pd
import numpy as np


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):

        self.fit_intercept = fit_intercept
        self.coefficient = None
        self.intercept = None

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.column_stack((np.ones_like(y), X))
            b = np.linalg.solve(X.T @ X, X.T @ y)
            self.intercept, self.coefficient = b[0], b[1:]
        else:
            self.coefficient = np.linalg.solve(X.T @ X, X.T @ y)

    def predict(self, X):
        yhat = X @ self.coefficient
        if self.fit_intercept:
            yhat += np.ones_like(yhat) * self.intercept
        return yhat


def run():
    df = pd.DataFrame({'x': [4, 4.5, 5, 5.5, 6, 6.5, 7],
                       'w': [1, -3, 2, 5, 0, 3, 6],
                       'z': [11, 15, 12, 9, 18, 13, 16],
                       'y': [33, 42, 45, 51, 53, 61, 62]})
    X, y = df[['x', 'w', 'z']].values, df['y'].values
    lr = CustomLinearRegression(fit_intercept=False)
    lr.fit(X, y)
    print(lr.predict(X))


run()
