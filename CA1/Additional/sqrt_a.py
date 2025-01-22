import time

def factorial(num):
    result = 1
    for i in range(1, num + 1):
        result *= i
    return result

start_time = time.time()

n = 5000

sum = 0
for k in range(n):
    numerator = factorial(2 * k + 1)
    denominator = (2 ** (3 * k + 1)) * (factorial(k) ** 2)
    sum += numerator / denominator

end_time = time.time()

print("Sum of Series:", sum)
print(f"Execution time: {end_time - start_time} seconds")