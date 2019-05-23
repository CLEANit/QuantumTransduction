#!/usr/bin/env python

import pickle
import matplotlib.pyplot as plt
import numpy as np
import cmocean.cm as cm
import sys

d = pickle.load(open(sys.argv[1], 'rb'))
n = len(d[0])

d_cp = np.copy(d)

plotting_data = np.empty((len(d), len(d[0])))
plotting_data[0:,] = np.arange(len(d[0]))

colors = {}
for elem in np.arange(len(d[0])):
	colors[elem] = elem

for i in range(1, len(d)):
	for j, elem in enumerate(d[i]):
		parents = d[i][elem]
		colors[elem] = 0.5 * (colors[parents[0]] + colors[parents[1]])
		plotting_data[i][j] = colors[elem]

plt.imshow(plotting_data.T, cmap=cm.deep)
plt.show()
# print(d_cp)





