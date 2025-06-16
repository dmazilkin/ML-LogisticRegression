import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from src.logreg import MyLogReg

N = 100

def main():
    np.random.seed(42)
    X = pd.DataFrame(np.zeros((N, 2)) + 10 * np.random.rand(N, 2))
    Y = pd.Series(np.zeros(N))
    Y = (X[0] > X[1] - 2).astype(int)

    plt.scatter(X.to_numpy()[Y == 1][:, 0], X.to_numpy()[Y == 1][:, 1], color='blue')
    plt.scatter(X.to_numpy()[Y == 0][:, 0], X.to_numpy()[Y == 0][:, 1], color='red')
    plt.show()

    model = MyLogReg()
    model.fit(X, Y, verbose=True)

    Y_predicted = model.predict(X)
    plt.scatter(X.to_numpy()[Y_predicted == 1][:, 0], X.to_numpy()[Y_predicted == 1][:, 1], color='blue')
    plt.scatter(X.to_numpy()[Y_predicted == 0][:, 0], X.to_numpy()[Y_predicted == 0][:, 1], color='red')
    plt.show()


if __name__ == '__main__':
    main()