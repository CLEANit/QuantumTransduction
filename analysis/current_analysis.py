#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas

font = {'family' : 'CMU Sans Serif',
#         'weight' : 'light',
        'size'   : 12}

plt.rc('font', **font)
plt.rc('text', usetex=True)

current_data = np.loadtxt(sys.argv[1])
n_structures = int(sys.argv[2])
n_generations = current_data.shape[0] // n_structures

currents = []
purities = []
objectives = []

def expRunningAvg(current_data, alpha=0.9):
    avgs = [current_data[0]]
    for i, elem in enumerate(current_data[1:]):
        avgs.append(avgs[i] * alpha + (1 - alpha) * elem)
    return avgs

# for finding the best structure
for i in range(n_generations):
    gen_data = current_data[i*n_structures:(i+1)*n_structures,:]
    objs = ((gen_data[:,0] / (gen_data[:,0] + gen_data[:,1]))**2 + (gen_data[:,3] / (gen_data[:,2] + gen_data[:,3]))**2 + (gen_data[:,0] + gen_data[:,3])**2)**0.5
    index = np.argmax(objs)
    objectives.append(objs[index])
    purities.append([gen_data[index,0] / (gen_data[index,0] + gen_data[index,1]), gen_data[index,3] / (gen_data[index,2] + gen_data[index,3])])
    currents.append(gen_data[index,:])

# for averaging
# for i in range(n_generations):
#     gen_data = current_data[i*n_structures:(i+1)*n_structures,:]
#     objs = ((gen_data[:,0] / (gen_data[:,0] + gen_data[:,1]))**2 + (gen_data[:,3] / (gen_data[:,2] + gen_data[:,3]))**2 + (gen_data[:,0] + gen_data[:,3])**2)**0.5
#     objectives.append(np.mean(objs))
#     purities.append([np.mean(gen_data[:,0] / (gen_data[:,0] + gen_data[:,1])), np.mean(gen_data[:,3] / (gen_data[:,2] + gen_data[:,3]))])
#     currents.append([np.mean(gen_data[:, 0]), np.mean(gen_data[:, 1]), np.mean(gen_data[:, 2]), np.mean(gen_data[:, 3])])

currents = np.array(currents)
purities = np.array(purities)
objectives = np.array(objectives)
fig, axes = plt.subplots(1, 3, figsize=(15,5))

axes[0].plot(range(n_generations), objectives, color='g', alpha=0.1)
axes[0].plot(range(n_generations), expRunningAvg(objectives), color='g', alpha=0.9)
axes[0].grid(linestyle='--', linewidth=0.5)
axes[0].set_xlabel('Generation')
axes[0].set_ylabel('Objective Function')


axes[1].plot(range(n_generations), purities[:,0], color='r', alpha=0.1),
axes[1].plot(range(n_generations), expRunningAvg(purities[:, 0], alpha=0.9), color='r', label='$k\'$')
axes[1].plot(range(n_generations), purities[:,1], color='b', alpha=0.1),
axes[1].plot(range(n_generations), expRunningAvg(purities[:, 1], alpha=0.9), color='b', label='$k$')
axes[1].grid(linestyle='--', linewidth=0.5)
axes[1].set_xlabel('Generation')
axes[1].set_ylabel('Polarization')
# axes[1].set_aspect('equal', adjustable='box')
# axes[1].set_facecolor('k', alpha=0.2)
axes[1].legend()

axes[2].plot(range(n_generations), currents[:,0], 'r', alpha=0.1)
axes[2].plot(range(n_generations), expRunningAvg(currents[:,0]), 'r', label='lead 1: $k\'$')
axes[2].plot(range(n_generations), currents[:,1], 'b', alpha=0.1)
axes[2].plot(range(n_generations), expRunningAvg(currents[:,1]), 'b', label='lead 1: $k$')
axes[2].plot(range(n_generations), currents[:,2], 'r--', alpha=0.1)
axes[2].plot(range(n_generations), expRunningAvg(currents[:,2]), 'r--', label='lead 2: $k\'$')
axes[2].plot(range(n_generations), currents[:,3], 'b--', alpha=0.1)
axes[2].plot(range(n_generations), expRunningAvg(currents[:,3]), 'b--', label='lead 2: $k$')
axes[2].grid(linestyle='--', linewidth=0.5)
axes[2].set_xlabel('Generation')
axes[2].set_ylabel('Current [$e \pi^{-1} \hbar^{-1}$]')
# axes[2].set_aspect('equal', adjustable='box')

axes[2].legend()
plt.tight_layout()
plt.savefig('purity_current.pdf')
# plt.show()