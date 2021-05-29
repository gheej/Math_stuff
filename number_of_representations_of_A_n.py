#Problem: for given n, d find the number of representations c_{n, d} of the
#quiver A_n with the vector of dimentions (d, ..., d) up to an isomorphism.  
#
#The program solves the problem: for given d it finds the matrix A, such that
#c_{n, d} = w * A^n * v, where w = (1, 1, ..., 1), v = (1, 0, ..., 0).  
#
#The output for given d is the matrix A, and then one may insert n to get the
#sequence c_{1, d}, ..., c_{n, d}.
# 
#----------------------------------------------------------------------------

import numpy as np
import sympy as sp 
from sympy.utilities.iterables import partitions 
from itertools import chain 
from collections import Counter 
from copy import deepcopy


def change_type(a):
    if type(a) == dict:
        b = [([i]*a[i]) for i in a]
        return tuple(sorted(list(chain(*b)))[::-1])
    elif type(a) == tuple:
        return dict(Counter([i for i in a if i>0] ))

def get_gen_func(part, X):
    P = 1
    i = 1
    for m in part:
        for j in range(1, part[m] + 1):
            P *= sp.div(X[0]**(m + 1) - X[i]**(m + 1), X[0] - X[i], domain='ZZ')[0]
            i += 1
    return sp.Poly(P)

def find_the_matrix(D):
    PART = [part.copy() for part in partitions(D)]
    PART_list = [change_type(i) for i in PART]
    X = sp.symbols('x:10')
    A = []
    d = {PART_list[i]:i for i in range(len(PART))}
    for i in range(len(PART)):
        Poly = get_gen_func(PART[i], X)
        a = [0] * len(PART)
        for j in Poly.as_dict().keys():
            a[d[tuple([k for k in sorted(j) if k > 0][::-1])]] += 1
        A.append(deepcopy(a))
    A = sp.Matrix(A)
    return A.T, len(PART)


D = int(input('Enter d: '))
A, number_of_partitions = find_the_matrix(D)

sp.pprint(A)
show_sequence = input('Show first n terms for d={}? If yes, enter desired n, otherwise type "no": '.format(D))
if show_sequence == 'no':
    pass
else:
    N = int(show_sequence) 
    v = np.array(sp.Matrix([1] + [0] * (number_of_partitions-1)))
    w = np.array(sp.Matrix([1]*number_of_partitions).T)
    A = np.array(A)
    if N > 0:
        print((w @ v)[0, 0])
    for i in range(N-1):
        v = A @ v
        print((w @ v)[0, 0])
