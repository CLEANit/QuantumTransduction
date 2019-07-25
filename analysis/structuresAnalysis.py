#!/usr/bin/env python

import dill
import numpy as np
import matplotlib.pyplot as plt
import kwant
import cmocean
import progressbar
from matplotlib.colors import to_rgba
import matplotlib.patches as mpatches
import matplotlib.cm as pltcm

###############################################
# get the colormap and set the transparency
###############################################
cmap = pltcm.get_cmap('jet')
cmap._init()
alphas = np.abs(np.ones(cmap.N) * 0.6)
cmap._lut[:-3,-1] = alphas
###############################################

###############################################
# load calculation and get currents
###############################################
ga = dill.load(open('restart/ga.dill', 'rb'))

structures = ga.rankGenerationWithSquare()

best_structure = structures[0]
print('Got best structure, now calculating currents...')

energy_range = best_structure.getEnergyRange()
energies = np.linspace(energy_range[0], energy_range[1], 8)

J = kwant.operator.Current(best_structure.system)

currents_kp = []
currents_k = []

bar = progressbar.ProgressBar()
for energy in energies:
	smatrix = kwant.smatrix(best_structure.system, energy)
	positives = np.where(smatrix.lead_info[0].velocities <= 0)[0]
	momentas = smatrix.lead_info[0].momenta[positives] 
	K_prime_indices = positives[np.ma.masked_where(momentas < 0, momentas).mask] 
	K_indices = positives[np.ma.masked_where(momentas >= 0, momentas).mask] 
	print('KP ind', K_prime_indices)
	print('K ind', K_indices)


	print('Shape modes for wf:', kwant.wave_function(best_structure.system, energy)(0).shape)
	K_prime_wfs = kwant.wave_function(best_structure.system, energy)(0)[K_prime_indices,:]
	K_wfs = kwant.wave_function(best_structure.system, energy)(0)[K_indices,:]
	currents_kp.append(np.sum([J(wf) for wf in K_prime_wfs], axis=0))
	currents_k.append(np.sum([J(wf) for wf in K_wfs], axis=0))

current_K_prime = np.sum(np.array(currents_kp), axis=0)
current_K = np.sum(np.array(currents_k), axis=0)
###############################################

###############################################
# plot the K and K prime currents with the structure
###############################################

fig, ax = plt.subplots(1,1, figsize=(20, 10))
best_structure.visualizeSystem(args={'ax': ax, 'hop_lw' : 0.0})
stuff_before = ax.get_children()

im = kwant.plotter.current(best_structure.system, current_K_prime, cmap=cmap, ax=ax, linecolor='k', max_linewidth=8., min_linewidth=0.0)

for elem in stuff_before:
	elem.set_zorder(-10)

plt.savefig('valley_current_K_prime.pdf')

fig, ax = plt.subplots(1,1, figsize=(20, 10))
best_structure.visualizeSystem(args={'ax': ax, 'hop_lw' : 0.0})
stuff_before = ax.get_children()

im = kwant.plotter.current(best_structure.system, current_K, cmap=cmap, ax=ax, linecolor='k', max_linewidth=8., min_linewidth=0.0)

for elem in stuff_before:
	elem.set_zorder(-10)

plt.savefig('valley_current_K.pdf')
###############################################
# all done
