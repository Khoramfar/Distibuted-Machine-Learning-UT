import numpy as np
import time

x = np.load('data.npy')
y = np.load('labels.npy')

np.random.seed(42)

n = np.arange(x.shape[0])
np.random.shuffle(n)

x = x[n]
y = y[n]

s = int(0.2 * x.shape[0])

X_train = x[:-s]
X_test = x[-s:]
y_train = y[:-s]
y_test = y[-s:]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

start_time = time.time()

lr = 0.01
iters = 1000

w = np.zeros(X_train.shape[1])
b = 0

for _ in range(iters):
    z = np.dot(X_train, w) + b
    y_pred = sigmoid(z)
    dw = np.dot(X_train.T, (y_pred - y_train)) / X_train.shape[0]
    db = np.sum(y_pred - y_train) / X_train.shape[0]

    w -= lr * dw
    b -= lr * db

z_test = np.dot(X_test, w) + b
y_pred_test = sigmoid(z_test)
y_pred_cls = np.where(y_pred_test > 0.5, 1, 0)

accuracy = np.sum(y_pred_cls == y_test) / len(y_test)

print("Execution time:", time.time() - start_time, "seconds")
print("Accuracy:", accuracy)
