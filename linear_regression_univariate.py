import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Data Generating - Data Loading
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

# Evaluate the Model
Y_pre = w[0][0] + w[0][1] * X_test
plt.plot(X_test, Y_pre, 'r')
plt.plot(X_test, Y_test, 'b.')
plt.legend(['Estimated_values', 'Trues_values'])
# plt.show()

# Evaluation of Error
error = Y_pre - Y_test
print(f'mean_error: {np.mean(error)}')
print(f'std_error: {np.std(error)}')
plt.plot(error)
plt.show()