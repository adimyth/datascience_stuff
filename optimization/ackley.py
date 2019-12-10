'''
Ackley Function
---------------

Ackley Function proposed by David Ackley in 1987, is a non-convex function used as a performance test problem for optimization algorithms.
'''

import numpy as np
# scikit-optimizer return self.f(x, *self.args)
def ackley_func(x):
    y = x[1]
    x = x[0]
    term1 = np.sqrt(0.5*(x**2+y**2))
    term2 = 0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))
    term3 = -20*(np.exp(-0.2*term1))
    term4 = -np.exp(term2)
    return term3+term4+np.exp(1)+20

