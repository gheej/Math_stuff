from math import isqrt
from itertools import product
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


def farey_sequence(n: int):
    # Credits to https://en.wikipedia.org/wiki/Farey_sequence
    a, b, c, d = 0, 1, 1, n
    yield a, b
    while (c <= n):
        k = (n + b) // d
        a, b, c, d = c, d, k * c - a, k * d - b
        yield a, b

def q(a, b, c, d):
    return a * b + a * c + a * d + b * c + b * d + c * d

def weight(a, b, c, d):
    """Input: a \ge b \ge c \ge d
    Output: 4! = 24 if a > b > c > d
    otherwise divide to the stabilizer """
    p, q, r = a == b, b == c, c == d
    if not (p or q or r):
        return 24
    elif p and not q and not r:
        return 12
    elif not p and q and not r:
        return 12
    elif not p and not q and r:
        return 12
    elif p and q and not r:
        return 4
    elif p and not q and r:
        return 6
    elif not p and q and r:
        return 4
    else: # p and q and r
        return 1

""" Want to calculate the number of a, b, c, d > 0 \in \mathbb
Z, z \in \mathbb Z + \omega \mathbb Z, such that |z|^2 (ab +
ac + ad + bc + bd + cd) = N
Want to output a list w/ such numbers for all N < N_0
"""

N = 10000
Ans = [0] * (N + 1)
Z = [0] * (N + 1) # value of |z|^2
ABCD = [0] * (N + 1) # value of q(a,b,c,d)
sN = isqrt(N) # floor of sqtr N
Z1 = [a^2 + a * b + b^2 for a, b in farey_sequence(sN)]
Z1 = Counter(Z1)
for i in Z1:
    if i < len(Z):
        Z[i] = Z1[i]
# a \ge b \ge c \ge d
a = 1
while a <= (N - 3) // 3:
    b = 1
    while b <= min(a, (N - 2 * a - 1) // (a + 2)):
        c = 1
        while c <= min(b, (N - a * b - a - b) // (a + b + 1)):
            d = 1
            while d <= min(c, (N - a * b - a * c - b * c) // (a + b + c)):
                # a \ge b \ge c \ge d \ge 1
                ABCD[q(a, b, c, d)] += weight(a, b, c, d)
                d += 1
            c += 1
        b += 1
    a += 1

"""Now we apply the divisor convolution
If i * j \le N there are 3 possibilies:
    1) i \le sN and j \le sN
    2) i \le sN and j > sN
    3) i > sN and j \le sN



"""

for i, j in product(range(1, sN + 1), repeat = 2):
    Ans[i * j] += Z[i] * ABCD[j]
for i in range(1, sN + 1):
    for j in range(sN + 1, N//i + 1):
        Ans[i * j] += Z[i] * ABCD[j] + Z[j] * ABCD[i] 

# Moving average

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

x = np.arange(len(Ans))
y = np.array(Ans)

fig, ax = plt.subplots()
ax.plot(x, y, label = 'initial')
ax.plot(x, smooth(y,20), label = 'smoothed')

reg = LinearRegression(fit_intercept = False).fit(x.reshape(-1, 1), y)

ax.plot(x, reg.predict(x.reshape(-1, 1)), label = 'best line')

ax.legend()
plt.show()

print('The coeffitient of the line: ', reg.coef_)

















