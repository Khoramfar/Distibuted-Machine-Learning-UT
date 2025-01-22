from mpi4py import MPI
import numpy as np
import time

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

data = np.load('data.npy')
labels = np.load('labels.npy')

data_split = np.array_split(data, size)
labels_split = np.array_split(labels, size)

X_local = data_split[rank]
y_local = labels_split[rank]

np.random.seed(42)

n = np.arange(X_local.shape[0])
np.random.shuffle(n)

X_local = X_local[n]
y_local = y_local[n]

s = int(0.2 * X_local.shape[0])

X_train_local = X_local[:-s]
X_test_local = X_local[-s:]
y_train_local = y_local[:-s]
y_test_local = y_local[-s:]

lr = 0.01
iters = 1000
n_samples, n_features = X_train_local.shape
w = np.zeros(n_features)
b = np.zeros(1)  # For Bcast Problem

start_time = time.time()

for _ in range(iters):
    z = np.dot(X_train_local, w) + b[0]
    y_pred = sigmoid(z)
    dw_local = np.dot(X_train_local.T, (y_pred - y_train_local)) / n_samples
    db_local = np.sum(y_pred - y_train_local) / n_samples

    dw_total = np.zeros(n_features)
    db_total = np.zeros(1)

    comm.Reduce(dw_local, dw_total, op=MPI.SUM, root=0)
    comm.Reduce(db_local, db_total, op=MPI.SUM, root=0)

    if rank == 0:
        dw_total /= size
        db_total /= size
        w -= lr * dw_total
        b -= lr * db_total

    comm.Bcast(w, root=0)
    comm.Bcast(b, root=0)

end_time = time.time()
exec_time = end_time - start_time

z_test = np.dot(X_test_local, w) + b[0]
y_pred_test = sigmoid(z_test)
y_pred_cls = np.where(y_pred_test > 0.5, 1, 0)

local_accuracy = np.sum(y_pred_cls == y_test_local) / len(y_test_local)

all_accuracies = comm.gather(local_accuracy, root=0)
all_times = comm.gather(exec_time, root=0)

if rank == 0:
    total_end_time = time.time()
    total_exec_time = total_end_time - start_time
    print("Total execution time:", total_exec_time, "seconds")
    
    for i in range(size):
        print("Node", i, ": Local Accuracy =", all_accuracies[i], "Execution Time =", all_times[i], "seconds")
    
    avg_accuracy = np.mean(all_accuracies)
    avg_time = np.mean(all_times)
    
    print("Average Accuracy:", avg_accuracy)
    print("Average Execution Time per Node:", avg_time, "seconds")
