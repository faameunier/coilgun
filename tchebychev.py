import numpy as np

n = 6

z = []
delta = []

z.append(np.cos((1 / 2 / n - 1) * np.pi))
for k in range(2, n + 1):
    z.append(np.cos(((2 * k - 1) / 2 / n - 1) * np.pi))
    delta.append(z[k - 1] - z[k - 2])

sigma = np.mean(delta)
sigma_chap = 2 / (n - 1) * np.sin((n - 1) / 2 / n * np.pi)

print(z)
print(delta)
print(sigma)
print(sigma_chap)
print(sigma / sigma_chap)

# n / x_max * 2 / (n - 1) <= 1/(2f)
# x_max * 4 * f <= n -1
# x_max * 4 * f + 1 <= n

# 2 * x_max / (n - 1) <= 1/(2f)
# 4 * f * x_max + 1 <= n
