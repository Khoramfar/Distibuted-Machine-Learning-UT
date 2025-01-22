import math
import time

start_time = time.time()

n = 5000

sum = 0
fact_k = 1
fact_2k_plus_1 = 1

for k in range(n):
    if k > 0:
        fact_k *= k
        fact_2k_plus_1 *= (2 * k) * (2 * k + 1)

    power_of_2 = 2 ** (3 * k + 1)
    denominator = power_of_2 * (fact_k ** 2)
    sum += fact_2k_plus_1 / denominator

print("Sum of Series:", sum)
print(f"Execution time: {time.time() - start_time} seconds")
