import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Data Generating - Data Loading
np.random.seed(2)
N = 1000
X1 = np.random.randn(N, 1)  # mean=0 and standard_deviation=1
X2 = 2 + np.random.randn(N, 1)  # mean=2 and standard_deviation=1
X3 = 2 * np.random.randn(N, 1)  # mean=0 and standard_deviation=2
X4 = 2 + 2 * np.random.randn(N, 1)  # mean=2 and standard_deviation=2
Y = 0.5 - 3 * X1 + 4 * X2 + 3 * X3 - 0.05 * X4 + 0.5 * np.random.randn(N, 1)  # noise whit mean=0 and std=1

# Data Preparing
one = np.ones((N, 1))
X = np.hstack((one, X1, X2, X3, X4))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

X_T_X_train = np.matmul(X_train.T, X_train)
X_T_Y_train = np.matmul(X_train.T, Y_train)
w = np.matmul(np.linalg.inv(X_T_X_train), X_T_Y_train)
print(w)

# Evaluate the Model
Y_pre = np.matmul(X_test, w)
plt.plot(Y_pre, 'r-')
plt.plot(Y_test, 'b.')
plt.legend(['Estimated_values', 'Trues_values'])
plt.show()

# Evaluation of Error
error = Y_pre - Y_test
print(f'mean_error: {np.mean(error)}')
print(f'std_error: {np.std(error)}')
plt.plot(error)
plt.show()
