N = 100

import numpy as np
from sympy import symbols, igcd #greatest common divisor
from matplotlib import pyplot as plt

def pygcd(x, y):
    if (x, y) == (0, 0):
        return 0
    expr = ((x + y) // igcd(x, y)) % 3
    return expr == 0

weights = np.vectorize(pygcd)

x, y = np.arange(0, N + 1), np.arange(0, N + 1)
xv, yv = np.meshgrid(x, y)
size = 10 * weights(xv, yv)

fig, ax = plt.subplots()
ax.set(title = "Integer (x, y) such that $\\frac{x + y}{gcd(x, y)} \\vdots 3$")
ax.scatter(xv, yv, s = size) 
plt.show()
