import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(2)
N = 1000
X = np.random.randn(N, 1)
Y = 1 + 2 * X + 0.1 * np.random.randn(N, 1)

plt.plot(X, Y, '.')
# plt.show()

# Data preparation
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
one_train = np.ones((len(X_train), 1))
X_train = np.hstack((one_train, X_train))

# parameter estimation using LS method
den = 0
num = np.zeros((1, 2))
for i in range(len(X_train)):
    num += X_train[i] * Y_train[i]
    den += X_train[i] ** 2
w = num/den

print(w)

