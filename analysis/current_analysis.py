#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as plt
import subprocess

font = {'family' : 'CMU Serif',
#         'weight' : 'light',
        'size'   : 36}

plt.rc('font', **font)
plt.rc('text', usetex=True)

def expRunningAvg(current_data, alpha=0.9):
    avgs = [current_data[0]]
    for i, elem in enumerate(current_data[1:]):
        avgs.append(avgs[i] * alpha + (1 - alpha) * elem)
    return avgs

def getMaxCurrentsForFile(filename):

    current_data = np.loadtxt(filename)
    n_structures = int(sys.argv[1])
    n_generations = current_data.shape[0] // n_structures

    # for finding the best structure
    currents = []
    purities = []
    objectives = []

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
    return currents, purities, objectives

file_list = subprocess.check_output('find . -name "all_cs.dat"', shell=True).split()
all_cs, all_ps, all_objs = [], [], []
for file in file_list:
    cs, ps, objs = getMaxCurrentsForFile(file.decode('utf-8'))
    all_cs.append(cs)
    all_ps.append(ps)
    all_objs.append(objs)

all_cs = np.array(all_cs)
all_ps = np.array(all_ps)
all_objs = np.array(all_objs)

print(np.argmax(all_objs), np.max(all_objs))


# fig, axes = plt.subplots(1, 3, figsize=(30,10))

# axes[0].plot(range(n_generations), objectives, color='g', alpha=0.175, lw=5.0)
# axes[0].plot(range(n_generations), expRunningAvg(objectives), color='g', alpha=0.9, label='Running Average', lw=5.0)
# axes[0].grid(linestyle='--', linewidth=0.5)
# axes[0].set_xlabel('Generation')
# axes[0].set_ylabel('Objective Function')
# axes[0].legend()

# axes[1].plot(range(n_generations), purities[:,0], color='r', alpha=0.175, lw=5.0),
# axes[1].plot(range(n_generations), expRunningAvg(purities[:, 0]), alpha=0.9, lw=5.0, color='r', label='$k\'$')
# axes[1].plot(range(n_generations), purities[:,1], color='b', alpha=0.175, lw=5.0),
# axes[1].plot(range(n_generations), expRunningAvg(purities[:, 1]), alpha=0.9, color='b', lw=5.0, label='$k$')
# axes[1].grid(linestyle='--', linewidth=0.5)
# axes[1].set_xlabel('Generation')
# axes[1].set_ylabel('Purity')
# # axes[1].set_aspect('equal', adjustable='box')
# # axes[1].set_facecolor('k', alpha=0.2)
# axes[1].legend()

# axes[2].plot(range(n_generations), currents[:,0], 'r', alpha=0.1, lw=5.0)
# axes[2].plot(range(n_generations), expRunningAvg(currents[:,0]), 'r', lw =5.0, label='lead 1: $k\'$')
# axes[2].plot(range(n_generations), currents[:,1], 'b', alpha=0.1, lw=5.0)
# axes[2].plot(range(n_generations), expRunningAvg(currents[:,1]), 'b', lw=5.0, label='lead 1: $k$')
# axes[2].plot(range(n_generations), currents[:,2], 'r--', alpha=0.1, lw=5.0)
# axes[2].plot(range(n_generations), expRunningAvg(currents[:,2]), 'r--', lw=5.0, label='lead 2: $k\'$')
# axes[2].plot(range(n_generations), currents[:,3], 'b--', alpha=0.1, lw=5.0)
# axes[2].plot(range(n_generations), expRunningAvg(currents[:,3]), 'b--', lw=5.0, label='lead 2: $k$')
# axes[2].grid(linestyle='--', linewidth=0.5)
# axes[2].set_xlabel('Generation')
# axes[2].set_ylabel('Current [$e \pi^{-1} \hbar^{-1}$]')
# # axes[2].set_aspect('equal', adjustable='box')

# axes[2].legend()
# plt.tight_layout()
# plt.savefig('purity_current.pdf')
# plt.show()
