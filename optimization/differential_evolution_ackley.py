import numpy as np
from scipy.optimize import differential_evolution
from ackley import ackley_func

bounds = [(-5, 5), (-5, 5)]
result = differential_evolution(ackley_func, bounds=bounds, maxiter=100)
print(f"Result:\n {result}")
