import math
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if rank == 0:
    start_time = time.time()

n = 5000

part_size = n // size
start = rank * part_size
end = (rank + 1) * part_size

sum_core = 0
fact_k = math.factorial(start)
fact_2k_plus_1 = math.factorial(2*start + 1)

for k in range(start, end):
    if k > 0:
        fact_k *= k
        fact_2k_plus_1 *= (2 * k) * (2 * k + 1)

    power_of_2 = 2 ** (3 * k + 1)
    denominator = power_of_2 * (fact_k ** 2)
    sum_core += fact_2k_plus_1 / denominator

total_sum = comm.reduce(sum_core, op=MPI.SUM, root=0)

if rank == 0:
    print("Sum of Series:", total_sum)
    print(f"Execution time: {time.time() - start_time} seconds")
