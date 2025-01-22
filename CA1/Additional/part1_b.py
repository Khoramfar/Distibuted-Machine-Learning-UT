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
fact_k = 1
fact_2k_plus_1 = 1
power_of_2 = 2

for k in range(n):
    if k > 0:
        fact_k *= k
        fact_2k_plus_1 *= (2 * k) * (2 * k + 1)
        power_of_2 *= 8

    denominator = power_of_2 * (fact_k ** 2)
    sum += fact_2k_plus_1 / denominator

if rank == 0:
    print("Sum of Series:", sum)
    print(f"Execution time: {time.time() - start_time} seconds")
