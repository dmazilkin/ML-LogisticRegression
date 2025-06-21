import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from typing import Union

from src.arg_parser import ArgParser
from src.logreg import LogReg

N = 800

def example_linear(n_iter: int, learning_rate: float, metric: Union[None, str]) -> None:
    np.random.seed(42)
    X = pd.DataFrame(np.zeros((N, 2)) + 10 * np.random.rand(N, 2))
    Y = (X.loc[:, 0] > X.loc[:, 1] - 2).astype(int)

    model = LogReg(n_iter=n_iter, metric=metric, learning_rate=learning_rate, sgd_sample=0.1)
    model.fit(X, Y, verbose=100)
    Y_predicted = model.predict(X)

    figure, axis = plt.subplots(1, 2)
    axis[0].scatter(X.to_numpy()[Y == 1][:, 0], X.to_numpy()[Y == 1][:, 1], color='blue')
    axis[0].scatter(X.to_numpy()[Y == 0][:, 0], X.to_numpy()[Y == 0][:, 1], color='red')
    axis[0].set_title('Real data')
    axis[1].scatter(X.to_numpy()[Y_predicted == 1][:, 0], X.to_numpy()[Y_predicted == 1][:, 1], color='blue')
    axis[1].scatter(X.to_numpy()[Y_predicted == 0][:, 0], X.to_numpy()[Y_predicted == 0][:, 1], color='red')
    axis[1].set_title('Predicted data')
    plt.show()

def example_regularization(n_iter: int, learning_rate: float, metric: Union[None, str], reg: Union[None, str], l1_coef: Union[None, float], l2_coef: Union[None, float]) -> None:
    np.random.seed(42)
    X = pd.DataFrame(np.zeros((N, 2)) + 10 * np.random.rand(N, 2))
    Y = (X.loc[:, 0] > X.loc[:, 1] - 2).astype(int)
    X = pd.concat([X, X.loc[:, 1]**2, X.loc[:, 1]**3], axis=1)
    
    model = LogReg(n_iter=n_iter, learning_rate=learning_rate, metric=metric, reg=reg, l1_coef=l1_coef, l2_coef=l2_coef)
    model.fit(X, Y, verbose=100)
    Y_predicted = model.predict(X)

    figure, axis = plt.subplots(1, 2)
    axis[0].scatter(X.to_numpy()[Y == 1][:, 0], X.to_numpy()[Y == 1][:, 1], color='blue')
    axis[0].scatter(X.to_numpy()[Y == 0][:, 0], X.to_numpy()[Y == 0][:, 1], color='red')
    axis[0].set_title('Real data')
    axis[1].scatter(X.to_numpy()[Y_predicted == 1][:, 0], X.to_numpy()[Y_predicted == 1][:, 1], color='blue')
    axis[1].scatter(X.to_numpy()[Y_predicted == 0][:, 0], X.to_numpy()[Y_predicted == 0][:, 1], color='red')
    axis[1].set_title('Predicted data')
    plt.show()
    
def main():
    parser = ArgParser()
    arguments = parser.parse()

    if arguments['example'] == 'linear':
        example_linear(arguments['iter'], arguments['lr'], arguments['metric'])

    if arguments['example'] == 'regularization':
        example_regularization(arguments['iter'],  arguments['lr'], arguments['metric'], arguments['reg'], arguments['l1'], arguments['l2'])

if __name__ == '__main__':
    main()