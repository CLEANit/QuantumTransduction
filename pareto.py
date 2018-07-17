#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def is_pareto_efficient(costs):
    """
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)  # Remove dominated points
    return is_efficient


data = np.loadtxt('output/phase_space.dat')[:,2:4]
# print(data)

pareto = np.array(sorted(data[is_pareto_efficient(data)], key=lambda x: x[0]))
plt.plot(data[:,0], data[:,1], '.')
plt.plot(pareto[:,0], pareto[:,1])
plt.title('Pareto frontier for multivariate optimization')
plt.xlabel('Purity of spin-up current in lead 1')
plt.ylabel('Purity of spin-down current in lead 2')
plt.show()