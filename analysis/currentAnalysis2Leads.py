#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as plt
import subprocess


font = {'family' : 'CMU Serif',
#         'weight' : 'light',
        'size'   : 18}

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
        objs = ((gen_data[:,1] / (gen_data[:,0] + gen_data[:,1]))**2 + ((gen_data[:,0] + gen_data[:,1]) / gen_data[:,0])**2)**0.5
        index = np.argmax(objs)
        objectives.append(objs[index])
        purities.append([gen_data[index,0] / (gen_data[index,0] + gen_data[index,1]), gen_data[index,1] / (gen_data[index,0] + gen_data[index,1])])
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

n_generations = all_objs.shape[1]

# for plotting max
# trial_ind, struct_ind = np.where(all_objs == np.max(all_objs))

# objectives = all_objs[trial_ind[0]]
# objectives_stds = np.std(all_objs, axis=0)
# purities = all_ps[trial_ind[0]]
# currents = all_cs[trial_ind[0]]

# for plotting averages
objectives = np.mean(all_objs, axis=0)
purities = np.mean(all_ps, axis=0)
currents = np.mean(all_cs, axis=0)
objectives_stds = np.std(all_objs, axis=0)
purities_stds = np.std(all_ps, axis=0)
currents_stds = np.std(all_cs, axis=0)

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(30,10))
outer = gridspec.GridSpec(1, 3)

inner_left = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0], wspace=0.2, hspace=0.2)
inner_middle = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1])
inner_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[2])


axes_left = plt.Subplot(fig, inner_left[0])
axes_middle_top = plt.Subplot(fig, inner_middle[0])
axes_middle_bottom = plt.Subplot(fig, inner_middle[1])

axes_right_0 = plt.Subplot(fig, inner_right[0])
axes_right_1 = plt.Subplot(fig, inner_right[1])



axes_left.plot(range(n_generations), objectives, color='g', alpha=1.0, lw=5.0, label='Objective function')
# axes_left.plot(range(n_generations), expRunningAvg(objectives), color='g', alpha=0.9, label='Running Average', lw=5.0)
# axes_left.errorbar(range(n_generations), objectives, objectives_stds, linestyle=None)
axes_left.fill_between(range(n_generations), objectives - objectives_stds, objectives + objectives_stds, color='g', alpha=0.2)
axes_left.set_xlabel('Generation')
# axes_left.set_ylabel('Objective Function')
fig.add_subplot(axes_left)

axes_middle_top.plot(range(n_generations), purities[:,0], color='r', alpha=1.0, lw=5.0, label='$k\'$')
# axes_middle_top.errorbar(range(n_generations), purities[:,0], purities_stds[:,0], ecolor='r',linestyle=None, alpha=0.5, capsize=5, elinewidth=3, linewidth=0)
axes_middle_top.fill_between(range(n_generations), purities[:, 0] - purities_stds[:, 0], purities[:,0] + purities_stds[:,0], color='r', alpha=0.2)
# axes[1].plot(range(n_generations), expRunningAvg(purities[:, 0]), alpha=0.9, lw=5.0, color='r', label='$k\'$')
axes_middle_bottom.plot(range(n_generations), purities[:,1], color='b', alpha=1.0, lw=5.0, label='$k$')
# axes_middle_bottom.errorbar(range(n_generations), purities[:,1], purities_stds[:,1], ecolor='b',linestyle=None, alpha=0.5, capsize=5, elinewidth=3, linewidth=0)
axes_middle_bottom.fill_between(range(n_generations), purities[:, 1] - purities_stds[:, 1], purities[:,1] + purities_stds[:,1], color='b', alpha=0.2)
# axes[1].plot(range(n_generations), expRunningAvg(purities[:, 1]), alpha=0.9, color='b', lw=5.0, label='$k$')
plt.setp(axes_middle_top.get_xticklabels(), visible=False)

axes_middle_bottom.set_xlabel('Generation')

fig.add_subplot(axes_middle_top)
fig.add_subplot(axes_middle_bottom)
# axes[1].set_aspect('equal', adjustable='box')
# axes[1].set_facecolor('k', alpha=0.2)
# axes[1].legend()

axes_right_0.plot(range(n_generations), currents[:,0], 'r', alpha=1.0, lw=5.0, label='lead 1: $k\'$')
axes_right_0.fill_between(range(n_generations), currents[:,0] - currents_stds[:,0], currents[:,0] + currents_stds[:,0], color='r', alpha=0.2)
# axes[2].plot(range(n_generations), expRunningAvg(currents[:,0]), 'r', lw =5.0, label='lead 1: $k\'$')
# axes_right_0.set_ylim([0.02, 0.12])
plt.setp(axes_right_0.get_xticklabels(), visible=False)

# axes[2].plot(range(n_generations), expRunningAvg(currents[:,1]), 'b', lw=5.0, label='lead 1: $k$')
axes_right_1.plot(range(n_generations), currents[:,1], 'b', alpha=1.0, lw=5.0, label='lead 1: $k$')
# axes_right_1.set_ylim([0.0, 0.01])
axes_right_1.fill_between(range(n_generations), currents[:,1] - currents_stds[:,1], currents[:,1] + currents_stds[:,1], color='r', alpha=0.2)
axes_right_1.set_xlabel('Generation')

fig.add_subplot(axes_right_0)
fig.add_subplot(axes_right_1)

# axes[2].plot(range(n_generations), expRunningAvg(currents[:,3]), 'b--', lw=5.0, label='lead 2: $k$')
# axes[2].grid(linestyle='--', linewidth=0.5)
# axes[2].set_xlabel('Generation')
# axes[2].set_ylabel('Current [$e \pi^{-1} \hbar^{-1}$]')
# axes[2].set_aspect('equal', adjustable='box')
for axis in fig.get_axes():
    axis.set_xlim([0, 64])
    axis.grid(linestyle='--', linewidth=0.5)
    axis.legend()
# axes[2].legend()
plt.tight_layout()
plt.savefig('purity_current.pdf')
plt.show()
