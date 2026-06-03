from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class RegressionModel:

    def __init__(self):
        self.models = {
            "linear": LinearRegression(),
            "ridge": Ridge(alpha=1.0),
            "lasso": Lasso(alpha=0.1)
        }
        self.model = None

    def select_model(self, name):

        if name not in self.models:
            raise ValueError("Model not supported")

        self.model = self.models[name]

    def fit(self, X_train, y_train):

        if self.model is None:
            raise Exception("Model not selected")

        self.model.fit(X_train, y_train)

    def predict(self, X_test):

        if self.model is None:
            raise Exception("Model not selected")

        return self.model.predict(X_test)

    def metrics (self, y_true, y_pred, n_features=None):

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }
