#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import cmocean.cm as cm
import subprocess as sbp


out = [elem.decode('utf-8') for elem in sbp.check_output('find . -name chromosomes*', shell=True).split()]

ds = []
for fname in out:
    ds.append(np.loadtxt(fname))

all_avgs_r, all_stds_r = np.empty((ds[0].shape[1] // 2, len(ds))), np.empty((ds[0].shape[1] // 2, len(ds)))
all_avgs, all_stds = np.empty((ds[0].shape[1], len(ds))), np.empty((ds[0].shape[1], len(ds)))

all_scatter_data = []
all_scatter_data_avgs = []
all_scatter_data_stds = []
for i, d in enumerate(ds):
    
    scatter_data = []
    scatter_data_avgs = []
    scatter_data_stds = []

    for j in range(d.shape[1] // 2):
        scatter_data.append((d[:, 2*j], d[:, 2*j + 1]))
        scatter_data_avgs.append((np.mean(d[:, 2*j]), np.mean(d[:, 2*j + 1])))
        scatter_data_stds.append((np.std(d[:, 2*j]), np.std(d[:, 2*j + 1])))

        all_avgs_r[j][i] = np.mean(np.sqrt(d[:, 2*j]**2 + d[:, 2*j+1]**2))
        all_stds_r[j][i] = np.std(np.sqrt(d[:, 2*j]**2 + d[:, 2*j+1]**2))
    
    for j in range(d.shape[1]):
        all_avgs[j][i] = np.mean(d[:, j])
        all_stds[j][i] = np.std(d[:, j])
    
    all_scatter_data.append(scatter_data)
    all_scatter_data_avgs.append(scatter_data_avgs)
    all_scatter_data_stds.append(scatter_data_stds)

all_scatter_data = np.array(all_scatter_data)
all_scatter_data_avgs = np.array(all_scatter_data_avgs)
all_scatter_data_stds = np.array(all_scatter_data_stds)

x = np.arange(len(ds))

fig = plt.figure(figsize=(20,10))
outer = gridspec.GridSpec(1, 2)
inner = gridspec.GridSpecFromSubplotSpec(ds[0].shape[1], 1, subplot_spec=outer[0])

for i in range(all_avgs.shape[0]):
    ax = plt.Subplot(fig, inner[i])
    ax.plot(x, all_avgs[i, :])
    ax.errorbar(x, all_avgs[i, :], fmt='.', yerr=all_stds[i, :])
    fig.add_subplot(ax)

ax = plt.Subplot(fig, outer[1])
for i, (sav, sd) in enumerate(zip(all_scatter_data_avgs, all_scatter_data_stds)):
    ax.scatter(sav[:, 0], sav[:, 1], c=cm.deep(i / all_scatter_data.shape[0]))
    ax.errorbar(sav[:, 0], sav[:, 1], fmt='.', yerr=sd[:, 1], xerr=sd[:, 0], c=cm.deep(i / all_scatter_data.shape[0] + 0.1))
fig.add_subplot(ax)

ax.legend(['Generation ' + str(i) for i in range(len(ds))])


plt.show() 




