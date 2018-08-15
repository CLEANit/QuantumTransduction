#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, griddata
import sys
from mpl_toolkits.mplot3d import Axes3D


def is_pareto_efficient(costs):
    """
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]>=c, axis=1)  # Remove dominated points
    return is_efficient


data = np.random.random((100, 3))

pareto = np.array(sorted(data[is_pareto_efficient(data)], key=lambda x: x[0]))

interp = interp2d(pareto[:,0], pareto[:,1], pareto[:,2])
x = np.linspace(np.min(data[:,0]), np.max(data[:,0]), 512)
y = np.linspace(np.min(data[:,1]), np.max(data[:,1]), 512)
X, Y = np.meshgrid(x,y)
fig = plt.figure()
ax = fig.gca(projection='3d')

pareto_X, pareto_Y = np.meshgrid(pareto[:,0], pareto[:,1])
pareto_Z = griddata((pareto[:,0], pareto[:,1]), pareto[:,2], (pareto_X, pareto_Y))
surf = ax.plot_surface(pareto_X, pareto_Y, pareto_Z)
scat = ax.scatter(data[:,0], data[:,1], data[:,2])
plt.show()
print(interp(data[:,0], data[:,1]).shape)
# zd = (interp(x, y)[:,None] - data[:,2])**2
# xd = (x[:,None] - data[:,0])**2
# r = np.sqrt(xd + yd)
# rmins = np.min(r, axis=0)
# closest_50 = np.argsort(rmins)[:50]

# plt.plot(data[:,0], data[:,1], '.')
# plt.plot(pareto[:,0], pareto[:,1])
# plt.plot(x, interp(x), '--')
# plt.plot(data[closest_50, 0], data[closest_50, 1], '.')
# plt.title('Pareto frontier for multivariate optimization')
# plt.xlabel('Purity of spin-up current in lead 1')
# plt.ylabel('Purity of spin-down current in lead 2')
# plt.show()
