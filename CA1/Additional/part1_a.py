import math
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    start_time = time.time()

n = 5000
sum = 0
for k in range(n):
    numerator = math.factorial(2 * k + 1)
    denominator = (2**(3*k + 1)) * (math.factorial(k)**2)
    value = numerator / denominator
    sum += value

if rank == 0:
    print("Sum of Series:", sum)
    print(f"Execution time: {time.time() - start_time} seconds")
