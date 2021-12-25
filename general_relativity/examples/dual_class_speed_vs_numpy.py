# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 21:50:51 2021

@author: TD
"""
import timeit

import matplotlib.pyplot as plt
import numpy as np

from FANTASYRPG import dual


func = lambda x: x**9 - np.sqrt(np.pi / 2) * x ** 4 + x - 10 + x * np.log(x)
dfunc = lambda x: 9 * x ** 8 - 4 * np.sqrt(np.pi / 2) * x ** 3 + 1 + np.log(x) + 1

ns = [10, 10 ** 2, 10 ** 4, 10 ** 6, 10 ** 7, 10 ** 8]
times_np = []
times_dual = []
for n in ns:
    x = np.linspace(-100, 100, n)
    
    # numpy
    s = timeit.default_timer()
    res_np = dfunc(x)
    e = timeit.default_timer()
    times_np.append(e-s)
    del e, s
    
    # dual
    s = timeit.default_timer()
    funcdual = func(dual(x, 1))
    res_dual = funcdual.s
    e = timeit.default_timer()
    times_dual.append(e-s)
    del e, s, funcdual
    
    print(
        f'{(res_np - res_dual).sum() < 1e-8}'
        ' -- numpy and dual dirivative agree to less than sum difference 1e-8'
    )
    del res_np, res_dual, x


plt.figure(figsize=(9,9))
plt.grid()
plt.plot(ns, times_np, label='numpy')
plt.plot(ns, times_dual, label='dual')
plt.plot(ns, 1e-7 * np.array(ns) ** 1.06, linestyle='--', color='grey', label='c*O(n)')
plt.xlabel('array size (n)')
plt.ylabel('evaluation time (sec)')
plt.legend()
plt.title(
    'Derivative Evaluation Time: Vectorized (numpy) vs Auto-differentiation\n'
    'test function: x**9 - np.sqrt(np.pi / 2) * x ** 4 + x - 10 + x * np.log(x)'
)
plt.show()