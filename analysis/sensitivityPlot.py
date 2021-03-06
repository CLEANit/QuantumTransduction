#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt 
import cmocean.cm as cm

font = {'family': 'CMU Serif',
        'weight' : 'light',
        'size' : 45 }
plt.rc('font', **font)
plt.rc('text', usetex=True)

sens = np.loadtxt('currents_vs_flips.dat')
sens_e = np.loadtxt('currents_vs_flips_edges.dat')

fig = plt.figure(figsize=(20,16))

plt.plot(sens[:, 0], sens[:, 1] / (sens[:, 1] + sens[:, 2]), c=cm.deep(0.25), lw=5, label='$K\'$ Purity - Random')
plt.plot(sens[:, 0], sens[:, 4] / (sens[:, 3 ] + sens[:, 4]), c=cm.deep(0.75), lw=5, label='$K$ Purity - Random')

plt.plot(sens_e[:, 0], sens_e[:, 1] / (sens_e[:, 1] + sens_e[:, 2]), '--', lw=5, c=cm.deep(0.25), label='$K\'$ Purity - Edges')
plt.plot(sens_e[:, 0], sens_e[:, 4] / (sens_e[:, 3] + sens_e[:, 4]), '--', lw=5, c=cm.deep(0.75), label='$K$ Purity - Edges')

plt.xlabel('Probability of Defect')
plt.ylabel('Purity')
plt.xlim([0.01, 1.0])
plt.ylim([0.01, 1.0])
plt.grid(linestyle='-.')
plt.legend()
plt.tight_layout()
plt.savefig('sensitivity.pdf')
