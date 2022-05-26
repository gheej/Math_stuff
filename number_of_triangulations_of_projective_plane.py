N = 100000

from math import isqrt
from itertools import product, chain
from collections import Counter

import numpy as np
import itertools
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from numpy.lib.stride_tricks import sliding_window_view
from sympy.ntheory import factorint


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

def partial_sum(a):
    b = np.zeros_like(a)
    s = 0
    for i in range(len(a)):
        s += a[i]
        b[i] = s
    return b

def moving_average(y, window_shape):
    return np.sum(sliding_window_view(y, window_shape), axis = 1)/window_shape

def moving_max(y, window_shape):
    b = sliding_window_view(y, window_shape)
    ind_b = sliding_window_view(np.arange(len(y)), window_shape)
    c = np.argmax(b, axis = 1)
    d = [i for i, j in itertools.groupby(ind_b[[tuple(i) for i in np.vstack((np.arange(len(c)), c))]])]
    return d

def moving_min(y, window_shape):
    b = sliding_window_view(y, window_shape)
    ind_b = sliding_window_view(np.arange(len(y)), window_shape)
    c = np.argmin(b, axis = 1)
    d = [i for i, j in itertools.groupby(ind_b[[tuple(i) for i in np.vstack((np.arange(len(c)), c))]])]
    return d

def find_quadr_alpha(y):
    squares = np.arange(len(y))**2
    alpha = np.linalg.lstsq(squares[len(y)//2:].reshape(-1, 1), y[len(y)//2:], rcond = None)[0][0]
    return alpha

def factor(y):
    ans = []
    for a in y:
        b = factorint(a)
        ans.append(list(chain(*[[i]*b[i] for i in b])))
    return ans

"""Now we apply the divisor convolution
If i * j \le N there are 3 possibilies:
    1) i \le sN and j \le sN
    2) i \le sN and j > sN
    3) i > sN and j \le sN
"""
def convolution(Z, ABCD):
    assert len(Z) == len(ABCD)
    N = len(Z) - 1
    sN = isqrt(N) # floor of sqtr N
    Ans = np.zeros_like(Z)
    for i, j in product(range(1, sN + 1), repeat = 2):
        Ans[i * j] += Z[i] * ABCD[j]
    for i in range(1, sN + 1):
        for j in range(sN + 1, N//i + 1):
            Ans[i * j] += Z[i] * ABCD[j] + Z[j] * ABCD[i] 
    return Ans

""" Want to calculate the number of a, b, c, d > 0 \in \mathbb
Z, z \in \mathbb Z + \omega \mathbb Z, such that |z|^2 (ab +
ac + ad + bc + bd + cd) = N
Want to output a list w/ such numbers for all N < N_0
"""

sN = isqrt(N) # floor of sqtr N
Z1_0 = Counter([a**2 - a * b + b**2 for a, b in farey_sequence(sN) if (a + b)%3 == 0])
Z1_12 = Counter([a**2 - a * b + b**2 for a, b in farey_sequence(sN) if (a + b)%3 != 0][:-1])

Z_0 = [0] * (N + 1) # value of |z|^2
Z_12 = [0] * (N + 1) # value of |z|^2
for i in Z1_0:
    if i < len(Z_0):
        Z_0[i] = Z1_0[i]
for i in Z1_12:
    if i < len(Z_12):
        Z_12[i] = Z1_12[i]

ABCD_0 = [0] * (N + 1) # value of q(a,b,c,d)
ABCD_12 = [0] * (N + 1) # value of q(a,b,c,d)
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
                ABCD_0[q(a, b, c, d)] += weight(a, b, c, d)
                if a%3 == b%3 and b%3 == c%3 and c%3 == d%3:
                    ABCD_12[q(a, b, c, d)] += weight(a, b, c, d)
                d += 1
            c += 1
        b += 1
    a += 1


Ans = [0] * (N + 1)
Ans = convolution(Z_0, ABCD_0) +  convolution(Z_12, ABCD_12)

# print(np.vstack((Z_0[:40], Z_12[:40], np.arange(40))).T)

fig, ax = plt.subplots()
 
x = np.arange(len(Ans))
y = np.array(Ans)
 
y = partial_sum(y)
# ax.set(title = '$\\#\\{|z|^2 (ab + ac + ad + bc + bd + cd) \\leq N\\} - \\alpha N^2$', xlabel = 'N', ylabel = 'Number of solutions')
# 
alpha = 0.20874321250560157071750716031497138622997487996283

"""I want to find alpha such that alpha * N^2
approximates my y the best
Use np.linalg.lstsq for that
to solve y = squares * alpha"""

# alpha = find_quadr_alpha(y)
u = (- 2.25 * y + alpha * x**2) / x**(3/2)
ax.plot(x, u)
ax.plot(x, np.ones_like(u) * u[-1])
# ax.plot(x, (2.25 * y - alpha * x**2) / x**(3/2), label ='$\\#\\{ \\frac{2}{3} |z|^2 (ab + ac + ad + bc + bd + cd) \\leq N\\} / N^2$')
# ax.plot(x, (2.25 * y) / x**2, label ='$f(n) / n^2$')
# ax.plot(x, np.ones_like(x) * alpha, label = str(round(alpha, 4)))
# ax.plot(x, 2.25 * y, label = 'Number of triangulations')
ax.set(xlabel = 'n', title = '$(f(n) - C n^2) / n^{\\frac{3}{2}} \\longrightarrow $' +  str(round(u[-1], 4)))
# print(alpha)

# ax.plot(x, alpha * x**2, label = 'best quardatic ' + str(round(alpha, 4)) + ' N^2')
#  print(np.vstack((x, y, ABCD, Z))[:, :50].T)
# 
# reg = LinearRegression(fit_intercept = False).fit(x.reshape(-1, 1), y)
# print('The coeffitient of the line: ', *reg.coef_)
# 
# window = 200
# min_ind = moving_min(y, window)
# max_ind = moving_max(y, window)
# 
# 
# ax.plot(x[window//2:-window//2+1], moving_average(y, window), label = 'local average')
# print("mins: ", *min_ind)
# ax.plot(min_ind, y[min_ind], label = 'local min')
# print("maxs: ", *max_ind)
# ax.plot(max_ind, y[max_ind], label = 'local max')
# 
# 
# print("!!!")
# for i in factor(min_ind):
#     print(*i)
# print("!!!!")
# for i in factor(max_ind):
#     print(*i)
# 
# const = 50
# 
# ax2.hist([i for i in list(chain(*factor(min_ind))) if i < const], bins = const)
# ax2.set(title = 'mins')
# ax3.hist([i for i in list(chain(*factor(max_ind))) if i < const], bins = const)
# ax2.set(title = 'maxs')


ax.legend()
plt.show()
