import math
import time

start_time = time.time()

n = 5000

sum = 0
for k in range(n):
    numerator = math.factorial(2 * k + 1)
    denominator = (2 ** (3 * k + 1)) * (math.factorial(k) ** 2)
    sum += numerator / denominator

end_time = time.time()

print("Sum of Series:", sum)
print(f"Execution time: {end_time - start_time} seconds")
