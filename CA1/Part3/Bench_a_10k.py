import numpy as np
import time
N = 10000

np.random.seed(42)
matrix = np.random.rand(N, N)

start_time = time.time()
np.dot(matrix, matrix)
print(f"Matrix multiplication time ({N}x{N}): {time.time() - start_time} seconds")
