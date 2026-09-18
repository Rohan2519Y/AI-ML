import numpy as np

x1 = np.array([1, 2, 3, 4])
x2 = np.array([4, 5, 8, 2])
Y = np.array([1, 6, 8, 12])

X = np.hstack((np.ones((4, 1)), x1.reshape((4, 1)), x2.reshape((4, 1))))
print(X)

A = (np.linalg.inv((X.T) @ X) @ (X.T)) @ Y
print(np.round(A, 3))

y = A[0] + (A[1] * x1) + (A[2] * x2)
print(y)