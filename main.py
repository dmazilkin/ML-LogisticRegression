import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from src.logreg import MyLogReg

N = 100

def main():
    np.random.seed(42)
    X = pd.DataFrame(np.zeros((N, 2)) + 10 * np.random.rand(N, 2))
    Y = (X[0] > X[1] - 2).astype(int)

    model = MyLogReg()
    model.fit(X, Y, verbose=True)
    Y_predicted = model.predict(X)

    figure, axis = plt.subplots(1, 2)
    axis[0].scatter(X.to_numpy()[Y == 1][:, 0], X.to_numpy()[Y == 1][:, 1], color='blue')
    axis[0].scatter(X.to_numpy()[Y == 0][:, 0], X.to_numpy()[Y == 0][:, 1], color='red')
    axis[0].set_title('Real data')
    axis[1].scatter(X.to_numpy()[Y_predicted == 1][:, 0], X.to_numpy()[Y_predicted == 1][:, 1], color='blue')
    axis[1].scatter(X.to_numpy()[Y_predicted == 0][:, 0], X.to_numpy()[Y_predicted == 0][:, 1], color='red')
    axis[1].set_title('Predicted data')
    plt.show()

if __name__ == '__main__':
    main()