import pandas as pd
import numpy as np
from typing import Union

EPS = 10 ** (-15)

class MyLogReg:
    def __init__(self, n_iter=10, learning_rate=0.1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None

    def __str__(self):
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X_input: pd.DataFrame, Y_input: pd.Series, verbose: Union[bool, int] = False):
        samples_count = X_input.shape[0]
        bias_feature = pd.DataFrame(np.zeros((samples_count, 1)) + 1)
        X = pd.concat([bias_feature, X_input], axis=1).to_numpy()
        Y_input = Y_input.to_numpy().reshape(-1, 1)
        features_count = X.shape[1]
        self.weights = np.zeros((features_count, 1)) + 1

        epoch = 0
        while epoch < self.n_iter:
            Y_predicted = self._predict(X, self.weights)
            loss = self._calc_loss(Y_input, Y_predicted)
            gradient = self._calc_gradient(Y_input, Y_predicted, X)
            self.weights -= self.learning_rate * gradient

            if (verbose > 0) and (epoch % verbose == 0):
                ind_log = 'start' if epoch == 0 else epoch
                print(f'ind_log | loss: {loss}')

            epoch += 1

    def _predict(self, X: np.array, weights: np.array) -> np.array:
        return self._sigmoid(X @ weights)

    def _sigmoid(self, Y_linear: np.array) -> np.array:
        Y = np.zeros(Y_linear.shape)
        Y[Y_linear >= 0] = 1 / (1 + np.exp(-Y_linear[Y_linear >= 0]))
        Y[Y_linear < 0] = np.exp(Y_linear[Y_linear < 0]) / (1 + np.exp(Y_linear[Y_linear < 0]))
        return Y

    def _calc_loss(self, Y_input: np.array, Y_predicted: np.array) -> np.array:
        return -1 * np.mean(Y_input * np.log(Y_predicted + EPS) + (1 - Y_input) * np.log(1 - Y_predicted + EPS))

    def _calc_gradient(self, Y_input: np.array, Y_predicted: np.array, X: np.array) -> np.array:
        return X.T @ (Y_predicted - Y_input) / X.shape[0]

    def get_coef(self):
        return self.weights

    def predict(self, X_input: pd.DataFrame):
        samples_count = X_input.shape[0]
        bias_feature = pd.DataFrame(np.zeros((samples_count, 1)) + 1)
        X = pd.concat([bias_feature, X_input], axis=1).to_numpy()
        return pd.Series((X @ self.weights >= 0.5).astype(int).ravel())