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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import axes3d
import tinyarray as ta
###############################################
# get the colormap and set the transparency
###############################################
cmap_r = pltcm.get_cmap('Reds')
cmap_r._init()
alphas = np.abs(np.ones(cmap_r.N) * 0.6)
cmap_r._lut[:-3,-1] = alphas

cmap_b = pltcm.get_cmap('Blues')
cmap_b._init()
alphas = np.abs(np.ones(cmap_b.N) * 0.6)
cmap_b._lut[:-3,-1] = alphas
###############################################

###############################################
# load calculation and get currents
###############################################
ga = dill.load(open('restart/ga.dill', 'rb'))

structures = ga.rankGenerationWithSquare()

best_structure = structures[0]
print('Got best structure, now calculating currents...')

energy_range = best_structure.getEnergyRange()
energies = np.linspace(energy_range[0], energy_range[1], 2)

J = kwant.operator.Current(best_structure.system)

currents_kp = []
currents_k = []

bar = progressbar.ProgressBar()
for energy in bar(energies):
	smatrix = kwant.smatrix(best_structure.system, energy)
	positives = np.where(smatrix.lead_info[0].velocities <= 0)[0]
	momentas = smatrix.lead_info[0].momenta[positives] 
	K_prime_indices = positives[np.ma.masked_where(momentas < 0, momentas).mask] 
	K_indices = positives[np.ma.masked_where(momentas >= 0, momentas).mask] 

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

fig = plt.figure(figsize=(20, 10))
ax = fig.gca(projection='3d')

tags = []
positions = []
sites = []
for s, v in best_structure.pre_system.site_value_pairs():
    # if the site is in the body
    # if self.body(s.pos):
    tags.append(s.tag)
    positions.append(s.pos)
    sites.append(s)
    # print (s.tag)
tags = np.array(tags)
positions = np.array(positions)
min_tag_sx = np.min(tags[:,0])
min_tag_sy = np.min(tags[:,1])
min_pos_sx = np.min(positions[:,0])
min_pos_sy = np.min(positions[:,1])
max_pos_sx = np.max(positions[:,0])
max_pos_sy = np.max(positions[:,1])

tag_length = np.max(tags[:,0]) - min_tag_sx
tag_width = np.max(tags[:,1]) - min_tag_sy

x = np.linspace(min_pos_sx, max_pos_sx, tag_length)
y = np.linspace(min_tag_sy, max_pos_sy, tag_width)
X, Y = np.meshgrid(x, y)

ax.plot_surface(X, Y, current_K.reshape(tag_length, tag_width))

# best_structure.visualizeSystem(args={'ax': ax})

# stuff_before = ax.get_children()
plt.show()
# im = kwant.plotter.current(best_structure.system, current_K_prime, cmap=cmap_r, ax=ax, linecolor='r', max_linewidth=8., min_linewidth=0.0)

# for elem in stuff_before:
# 	elem.set_zorder(-10)

# divider = make_axes_locatable(ax)
# cax = divider.append_axes('right', size='2.5%', pad=0.05)
# fig.colorbar(im, ax=ax, cax=cax, orientation='vertical')

# plt.savefig('valley_current_K_prime_3D.pdf')

# fig, ax = plt.subplots(1,1, figsize=(20, 10))
# best_structure.visualizeSystem(args={'ax': ax})
# stuff_before = ax.get_children()

# im = kwant.plotter.current(best_structure.system, current_K, cmap=cmap_b, ax=ax, linecolor='b', max_linewidth=8., min_linewidth=0.0)

# for elem in stuff_before:
# 	elem.set_zorder(-10)

# divider = make_axes_locatable(ax)
# cax = divider.append_axes('right', size='2.5%', pad=0.05)
# fig.colorbar(im, ax=ax, cax=cax, orientation='vertical')

# plt.savefig('valley_current_K_3D.pdf')
# ###############################################
# # all done
