import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import sigma
from sklearn.model_selection import train_test_split

# Linear regression multivariate non-iid

# Data Generating - Data Loading
np.random.seed(2)
N = 1000
X1 = np.random.randn(N, 1)  # mean=0 and standard_deviation=1
X2 = 2 + np.random.randn(N, 1)  # mean=2 and standard_deviation=1
X3 = 2 * np.random.randn(N, 1)  # mean=0 and standard_deviation=2
X4 = 2 + 2 * np.random.randn(N, 1)  # mean=2 and standard_deviation=2
Y = np.empty((N, 1))
for k in range(N):
    sigma = np.random.randint(8)
    if sigma == 0: #because if std=0, then there is noise
        sigma = 1
    Y[k] = 0.5 - 3 * X1[k] + 4 * X2[k] + 3 * X3[k] - 0.05 * X4[k] + sigma * np.random.randn(1, 1)  # noise whit mean=0 and std=sigma

# plt.plot(Y)
# plt.show()

# Data Preparing
one = np.ones((N, 1))
X = np.hstack((one, X1, X2, X3, X4))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# calculating the Q matrix (weight matrix)
distance = np.exp(np.abs((X_train - np.mean(X_train))))
mean = np.mean(distance, axis=1).reshape(-1, 1)
weights = np.diagflat(np.matrix(mean))
X_T_Q_train = np.matmul(X_train.T, np.linalg.inv(weights))
X_T_Q_X_train = np.matmul(X_T_Q_train, X_train)
X_T_Q_Y_train = np.matmul(X_T_Q_train, Y_train)
w = np.matmul(np.linalg.inv(X_T_Q_X_train), X_T_Q_Y_train)
print(w)
#
# Evaluate the Model
Y_pre = np.matmul(X_test, w)
plt.plot(Y_pre, 'r.')
plt.plot(Y_test, 'bo')
plt.legend(['Estimated_values', 'Trues_values'])
# plt.show()


# Evaluation of Error
error = Y_pre - Y_test
print(f'mean_error: {np.mean(error)}')
print(f'std_error: {np.std(error)}')
plt.plot(error)
plt.show()
