import pandas as pd
import numpy as np
import random
from typing import Union

EPS = 10 ** (-15)
THRESHOLD = 0.5

class LogReg:
    def __init__(self, n_iter: int = 10, learning_rate: float = 0.1, metric: Union[None, str] = None, reg: Union[None, str] = None, l1_coef: float = 0.0, l2_coef: float = 0.0, sgd_sample: Union[int, float, None] = None, random_state: int = 42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric
        self.best_metric = None
        self.weights = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self):
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X_input: pd.DataFrame, Y_input: pd.Series, verbose: Union[bool, int] = False):
        X = self._preprocess_features(X_input)
        Y_input = Y_input.to_numpy().reshape(-1, 1)
        features_count = X.shape[1]
        self.weights = np.zeros((features_count, 1)) + 1
        epoch = 0
        
        if self.sgd_sample is not None:
            random.seed(self.random_state)

        while epoch < self.n_iter:
            Y_predicted = self._predict_proba(X, self.weights)
            loss = self._calc_loss(Y_input, Y_predicted)
            gradient = self._calc_gradient(Y_input, Y_predicted, X)
            lr = self.learning_rate(epoch + 1) if callable(self.learning_rate) else self.learning_rate
            self.weights -= lr * gradient

            if (verbose > 0) and (epoch % verbose == 0):
                ind_log = 'start' if epoch == 0 else epoch
                metric = f' | {self.metric} : {self._calc_metric(self.metric, Y_input, Y_predicted)}' if self.metric else ''
                print(f'{ind_log} | loss: {loss}' + metric)

            epoch += 1

            if epoch == self.n_iter and self.metric is not None:
                Y_predicted = self._predict_proba(X, self.weights)
                self.best_metric = self._calc_metric(self.metric, Y_input, Y_predicted)

    def _predict_proba(self, X: np.array, weights: np.array) -> np.array:
        return self._sigmoid(X @ weights)

    def _predict_class(self, Y_predicted_proba: np.array) -> np.array:
        return (Y_predicted_proba > THRESHOLD).astype(int)

    def _sigmoid(self, Y_linear: np.array) -> np.array:
        Y = np.zeros(Y_linear.shape)
        Y[Y_linear >= 0] = 1 / (1 + np.exp(-Y_linear[Y_linear >= 0]))
        Y[Y_linear < 0] = np.exp(Y_linear[Y_linear < 0]) / (1 + np.exp(Y_linear[Y_linear < 0]))
        return Y

    def _l1_reg(self, option: str) -> np.array:
        exclude_bias = np.zeros(self.weights.shape) + 1
        exclude_bias[0] = 0
        exclude_bias *= self.weights
        options = {
            'gradient': self.l1_coef * np.sign(exclude_bias),
            'loss': self.l1_coef * exclude_bias,
        }

        return options[option]

    def _l2_reg(self, option: str) -> np.array:
        exclude_bias = np.zeros(self.weights.shape) + 1
        exclude_bias[0] = 0
        exclude_bias *= self.weights
        options = {
            'gradient': 2 * self.l2_coef * exclude_bias,
            'loss': self.l2_coef * exclude_bias ** 2,
        }

        return options[option]

    def _elastic_reg(self, option: str) -> np.array:
        return self._l1_reg(option) + self._l2_reg(option)

    def _calc_loss(self, Y_input: np.array, Y_predicted: np.array) -> np.array:
        available_regs = {
            'l1': self._l1_reg,
            'l2': self._l2_reg,
            'elasticnet': self._elastic_reg,
        }

        reg = np.sum(available_regs[self.reg]('loss')) if self.reg in available_regs else 0
        log_loss = -1 * np.mean(Y_input * np.log(Y_predicted + EPS) + (1 - Y_input) * np.log(1 - Y_predicted + EPS))
        return log_loss + reg

    def _calc_gradient(self, Y_input: np.array, Y_predicted: np.array, X: np.array) -> np.array:
        available_regs = {
            'l1': self._l1_reg,
            'l2': self._l2_reg,
            'elasticnet': self._elastic_reg,
        }

        reg_grad = available_regs[self.reg]('gradient') if self.reg in available_regs else 0
        loss_grad = self._calc_batch_gradient(Y_input, Y_predicted, X) if self.sgd_sample is None else self._calc_mini_batch_gradient(Y_input, Y_predicted, X)
        return loss_grad + reg_grad
    
    def _calc_batch_gradient(self, Y_input: np.array, Y_predicted: np.array, X: np.array) -> np.array:
        return X.T @ (Y_predicted - Y_input) / X.shape[0]
    
    def _calc_mini_batch_gradient(self, Y_input: np.array, Y_predicted: np.array, X: np.array) -> np.array:
        sgd_sample = self.sgd_sample if isinstance(self.sgd_sample, int) else round(Y_input.shape[0] * self.sgd_sample)
        batch_indices = random.sample(range(Y_input.shape[0]), sgd_sample)
        return X[batch_indices].T @ (Y_predicted[batch_indices] - Y_input[batch_indices]) / sgd_sample

    def _calc_metric(self, metric: str, Y_input: np.array, Y_predicted: np.array) -> float:
        available_metrics_class = {
            'accuracy': self._calc_accuracy,
            'precision': self._calc_precision,
            'recall': self._calc_recall,
            'f1': self._calc_f1,
        }

        available_metrics_probability = {
            'roc_auc': self._calc_roc_auc,
        }

        metric_method = available_metrics_class[metric] if metric in available_metrics_class else available_metrics_probability[metric]
        Y_predicted_preprocess = Y_predicted if metric in available_metrics_probability else self._predict_class(Y_predicted)

        return metric_method(Y_input, Y_predicted_preprocess)

    def _calc_tp(self, Y_input: np.array, Y_predicted: np.array) -> float:
        return np.sum((Y_predicted == 1) & (Y_input == 1))

    def _calc_fp(self, Y_input: np.array, Y_predicted: np.array) -> float:
        return np.sum(Y_input[Y_predicted == 1] == 0)

    def _calc_tn(self, Y_input: np.array, Y_predicted: np.array) -> float:
        return np.sum((Y_predicted == 0) & (Y_input == 0))

    def _calc_fn(self, Y_input: np.array, Y_predicted: np.array) -> float:
        return np.sum(Y_input[Y_predicted == 0] == 1)

    def _calc_accuracy(self, Y_input: np.array, Y_predicted: np.array) -> float:
        TP = self._calc_tp(Y_input, Y_predicted)
        TN = self._calc_tn(Y_input, Y_predicted)
        print(TP, TN, Y_predicted.size)
        return (TP + TN) / Y_predicted.size

    def _calc_precision(self, Y_input: np.array, Y_predicted: np.array) -> float:
        TP = self._calc_tp(Y_input, Y_predicted)
        FP = self._calc_fp(Y_input, Y_predicted)
        return TP / (TP + FP)

    def _calc_recall(self, Y_input: np.array, Y_predicted: np.array) -> float:
        TP = self._calc_tp(Y_input, Y_predicted)
        FN = self._calc_fn(Y_input, Y_predicted)
        return TP / (TP + FN)

    def _calc_f1(self, Y_input: np.array, Y_predicted: np.array) -> float:
        precision = self._calc_precision(Y_input, Y_predicted)
        recall = self._calc_recall(Y_input, Y_predicted)
        return 2 * precision * recall / (precision + recall)

    def _calc_roc_auc(self, Y_input: np.array, Y_predicted: np.array) -> float:
        N_count = np.sum(Y_input == 0)
        P_count = np.sum(Y_input == 1)
        roc_auc = 0
        for negative in Y_predicted[Y_input == 0]:
            smaller = np.sum(negative < Y_predicted[Y_input == 1])
            equal = 0.5 * np.sum(negative == Y_predicted[Y_input == 1])
            roc_auc += smaller + equal
        return roc_auc / N_count / P_count

    def get_coef(self):
        return self.weights
    
    def _preprocess_features(self, X_input: pd.DataFrame) -> np.array:
        samples_count = X_input.shape[0]
        bias_feature = pd.DataFrame(np.zeros((samples_count, 1)) + 1)
        X_preprocess = X_input.reset_index(drop=True)
        return pd.concat([bias_feature, X_preprocess], axis=1).to_numpy()

    def predict(self, X_input: pd.DataFrame) -> pd.Series:
        X = self._preprocess_features(X_input)
        return pd.Series(self._sigmoid(X @ self.weights).ravel() > THRESHOLD).astype(int)

    def predict_proba(self, X_input: pd.DataFrame) -> np.array:
        X = self._preprocess_features(X_input)
        return pd.Series(self._sigmoid(X @ self.weights).ravel())

    def get_best_score(self):
        return self.best_metric