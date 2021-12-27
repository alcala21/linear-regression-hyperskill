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

    def print(self):
        print({'Intercept': self.intercept, 'Coefficient': self.coefficient})


def run():
    df = pd.DataFrame({'y': [33, 42, 45, 51, 53, 61, 62],
                       'x': [4.0, 4.5, 5, 5.5, 6.0, 6.5, 7.0]})
    lr = CustomLinearRegression()
    lr.fit(df['x'].values, df['y'])
    lr.print()


run()
