#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as plt

font = {'family' : 'CMU Sans Serif',
#         'weight' : 'light',
        'size'   : 12}

plt.rc('font', **font)
plt.rc('text', usetex=True)

data = np.loadtxt(sys.argv[1])
n_structures = int(sys.argv[2])
n_generations = data.shape[0] // n_structures

all_means = []
purities = []

def expRunningAvg(data, alpha=0.9):
    avgs = [data[0]]
    for i, elem in enumerate(data[1:]):
        avgs.append(avgs[i] * alpha + (1 - alpha) * elem)
    return avgs


for i in range(n_generations):
    gen_data = data[i*n_structures:(i+1)*n_structures,:]
    purities.append((np.mean(gen_data[:,0] / (gen_data[:,0] + gen_data[:,1])), np.mean(gen_data[:,3] / (gen_data[:,2] + gen_data[:,3]))))
    means = [np.mean(gen_data[:,i]) for i in range(gen_data.shape[1])]
    all_means.append(means)

all_means = np.array(all_means)
purities = np.array(purities)

fig, axes = plt.subplots(1, 2, figsize=(10,5))

axes[0].plot(range(n_generations), purities[:,0], color='r', alpha=0.25),
axes[0].plot(range(n_generations), expRunningAvg(purities[:, 0], alpha=0.9), color='r', label='$k\'$')
axes[0].plot(range(n_generations), purities[:,1], color='b', alpha=0.25),
axes[0].plot(range(n_generations), expRunningAvg(purities[:, 1], alpha=0.9), color='b', label='$k$')
axes[0].grid(linestyle='--', linewidth=0.5)
axes[0].set_xlabel('Generation')
axes[0].set_ylabel('Polarization')
# axes[0].set_aspect('equal', adjustable='box')
# axes[0].set_facecolor('k', alpha=0.2)
axes[0].legend()

axes[1].plot(range(n_generations), all_means[:,0], 'r', alpha=0.25)
axes[1].plot(range(n_generations), expRunningAvg(all_means[:,0]), 'r', label='lead 1: $k\'$')
axes[1].plot(range(n_generations), all_means[:,1], 'b', alpha=0.25)
axes[1].plot(range(n_generations), expRunningAvg(all_means[:,1]), 'b', label='lead 1: $k$')
axes[1].plot(range(n_generations), all_means[:,2], 'r--', alpha=0.25)
axes[1].plot(range(n_generations), expRunningAvg(all_means[:,2]), 'r--', label='lead 2: $k\'$')
axes[1].plot(range(n_generations), all_means[:,3], 'b--', alpha=0.25)
axes[1].plot(range(n_generations), expRunningAvg(all_means[:,3]), 'b--', label='lead 2: $k$')
axes[1].grid(linestyle='--', linewidth=0.5)
axes[1].set_xlabel('Generation')
axes[1].set_ylabel('Current [$e \pi^{-1} \hbar^{-1}$]')
# axes[1].set_aspect('equal', adjustable='box')

axes[1].legend()
plt.tight_layout()
plt.show()