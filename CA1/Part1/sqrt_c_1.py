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

print("Rank: ", rank,"Start: ", start, "End: ", end)

sum_core = 0

for k in range(start, end):
    numerator = math.factorial(2 * k + 1)
    denominator = (2**(3*k + 1)) * (math.factorial(k)**2)
    value = numerator / denominator
    sum_core += value

total_sum = comm.reduce(sum_core, op=MPI.SUM, root=0)

if rank == 0:
    print("Sum of Series:", total_sum)
    print(f"Execution time: {time.time() - start_time} seconds")

