#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt 

font = {'family': 'CMU Serif',
        'weight' : 'light',
        'size' : 36 }
plt.rc('font', **font)
plt.rc('text', usetex=True)

sens = np.loadtxt('currents_vs_flips.dat')
sens_e = np.loadtxt('currents_vs_flips_edges.dat')

plt.plot(sens[:, 0], sens[:, 1] / (sens[:, 1] + sens[:, 2]), label='K\' Purity - Random')
plt.plot(sens[:, 0], sens[:, 4] / (sens[:, 3 ] + sens[:, 4]), label='K Purity - Random')

plt.plot(sens_e[:, 0], sens_e[:, 1] / (sens_e[:, 1] + sens_e[:, 2]), label='K\' Purity - Edges')
plt.plot(sens_e[:, 0], sens_e[:, 4] / (sens_e[:, 3] + sens_e[:, 4]), label='K Purity - Edges')

plt.xlabel('Probability of Defect')
plt.ylabel('Purity')

plt.legend()

plt.show()
