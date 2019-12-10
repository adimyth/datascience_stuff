import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

x = np.linspace(0, 10, 500)
y = np.cos(x) + np.random.normal(0, 0.2, 500)
# plt.scatter(x, y, c='r', s=1)
# plt.plot(x, np.cos(x), label='cos(x)')
# plt.legend()
# plt.show()

def fmodel(x, w):
    return w[0] + w[1]*x + w[2] * x**2 + w[3] * x**3 + w[4] * x**4 + w[5] * x**5


def rmse(w):
    y_pred = fmodel(x, w)
    return np.sqrt(sum((y - y_pred)**2) / len(y))

result = differential_evolution(rmse, bounds=[(-5, 5)]*6, maxiter=2000)
print(f"Result: {result}")
plt.scatter(x, y, c='r', s=1)
plt.plot(x, np.cos(x), label='cos(x)')
plt.plot(x, fmodel(x, result['x']), label='result', c='g')
plt.legend()
plt.show()
