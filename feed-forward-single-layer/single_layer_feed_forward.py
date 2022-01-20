import numpy as np

# Training data for NAND.
x = np.array([
    [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]
    ])
y = np.array([1, 1, 1, 0])
w = np.array([0.0, 0.0, 0.0])

eta = 0.5
for t in range(100):
    for i in range(len(y)):
        y_pred = np.heaviside(np.dot(x[i], w), 0)
        w += (y[i] - y_pred) * eta * x[i]

result = np.heaviside(np.dot(x, w), 0)