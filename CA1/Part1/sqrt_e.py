import math
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    start_time = time.time()

n = 5000

sum_core = 0

for k in range(rank, n, size):
    numerator = math.factorial(2 * k + 1)
    denominator = (2 ** (3 * k + 1)) * (math.factorial(k) ** 2)
    value = numerator / denominator
    sum_core += value

total_sum = comm.reduce(sum_core, op=MPI.SUM, root=0)

if rank == 0:
    end_time = time.time()
    print("Sum of Series:", total_sum)
    print(f"Execution time: {end_time - start_time} seconds")
