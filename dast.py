import numpy as np
import time

field = np.zeros((100, 100), dtype=np.int32)

start = time.time()
for r in range(11):
    for c in range(11):
        x = field[r, c]
delta1 = time.time() - start
print(f"For loop: {delta1}")

start = time.time()
x = field[:11, :11]
delta2 = time.time() - start
print(f"Slice: {delta2}")

print(f"Pecent diff = {abs(delta2 - delta1) / delta1}")
