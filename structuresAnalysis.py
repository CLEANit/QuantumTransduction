#!/usr/bin/env python

import dill
import numpy as np
import matplotlib.pyplot as plt
import kwant
import cmocean
from matplotlib.colors import to_rgba
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib
from functools import partial

from src.qmt.helper import pointInHull

font = {'family' : 'CMU Sans Serif',
#         'weight' : 'light',
        'size'   : 36}
plt.rc('font', **font)
plt.rc('text', usetex=True)

ga = dill.load(open('restart/ga.dill', 'rb'))

structures = ga.getCurrentGeneration()
purities = ga.current_vectors
scores = np.sqrt(purities[:,0]**2 + purities[:,1]**2)
best_score = np.argsort(scores)[-1]
print(purities[best_score])
print('Best score:', scores[best_score])
print('Best one can do is 2**0.5 =', 2**0.5)

best_structure = structures[best_score]
syst = best_structure.system
# best_structure.visualizeSystem()
# plt.show()

# currents_0_1 = best_structure.getValleyPolarizedCurrent(0, 1)
# currents_0_2 = best_structure.getValleyPolarizedCurrent(0, 2)
# print(currents_0_1, currents_0_2)
# es1, cs1 = best_structure.getConductance(0, 1)
# es2, cs2 = best_structure.getConductance(0, 2)

# plt.plot(es1, cs1, '-', color=plt.get_cmap('viridis')(0.25))
# plt.plot(es2, cs2, '--', color=plt.get_cmap('viridis')(0.75))
# plt.legend(['$k\'$', '$k$'])
# plt.xlabel('Energy [eV]')
# plt.ylabel('Conductance [$2e^2 \hbar^{-1}$]')
# plt.show()

J = kwant.operator.Current(syst)
energy = 1.0
smatrix = kwant.smatrix(syst, energy)
positives = np.where(smatrix.lead_info[0].velocities <= 0)[0]
momentas = smatrix.lead_info[0].momenta[positives]
K_prime_indices = np.where(momentas < 0)[0]
K_indices = np.where(momentas >= 0)[0]

K_prime_wfs = kwant.wave_function(syst, energy)(0)[K_prime_indices,:]
K_wfs = kwant.wave_function(syst, energy)(0)[K_indices,:]
current_K_prime = np.sum(J(wf) for wf in K_prime_wfs)
current_K = np.sum(J(wf) for wf in K_wfs)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

fig, ax = plt.subplots(2,1, figsize=(10, 10))
red_cm = truncate_colormap(cmocean.cm.amp, minval=0.0, maxval=1.0)
red_cm._init()
alphas = np.abs(np.linspace(1.0, 1.0, red_cm.N))
red_cm._lut[:-3,-1] = alphas

blue_cm = truncate_colormap(cmocean.cm.ice_r, minval=0.0, maxval=1.0)
blue_cm._init()
alphas = np.abs(np.linspace(1.0, 1.0, blue_cm.N))
blue_cm._lut[:-3,-1] = alphas

def siteColours(c, site):
    # print(list(self.pre_system.sites())[site])
    if pointInHull(site.pos, best_structure.hull):
        return 'k'
    else:
        return c

def hopColours(c, site1, site2):
    if pointInHull(site1.pos, best_structure.hull) and pointInHull(site2.pos, best_structure.hull):
        return 'k'
    else:
        return c

sfig, sax = plt.subplots(figsize=(25,10))

kwant.plot(best_structure.pre_system, 
           site_lw=0.1,
           lead_site_lw=0,
           site_size=0.0,
           site_color=partial(siteColours, to_rgba('g', 0.1)),
           hop_lw=0.1,
           hop_color=partial(hopColours, to_rgba('g', 0.5)),
           lead_color=to_rgba('r', 0.5),
           colorbar=False,
           show=False,
           ax=sax)
sax.axis('equal')
sax.axis('off')
plt.tight_layout()
sfig.savefig('structure.pdf')

kwant.plot(best_structure.pre_system, 
           site_lw=0.1,
           lead_site_lw=0,
           site_size=0.0,
           site_color=partial(siteColours, to_rgba('g', 0.1)),
           hop_lw=0.1,
           hop_color=partial(hopColours, to_rgba('g', 0.5)),
           lead_color=to_rgba('r', 0.5),
           colorbar=False,
           show=False,
           ax=ax[0])

kwant.plot(best_structure.pre_system, 
           site_lw=0.1,
           lead_site_lw=0,
           site_size=0.0,
           site_color=partial(siteColours, to_rgba('g', 0.1)),
           hop_lw=0.1,
           hop_color=partial(hopColours, to_rgba('g', 0.5)),
           lead_color=to_rgba('r', 0.5),
           colorbar=False,
           show=False,
           ax=ax[1])

kwant.plotter.current(syst, current_K_prime, cmap=red_cm, ax=ax[0], linecolor=cmocean.cm.amp(0.75), density=0.5, relwidth=0.1)
plt.colorbar(ax[0].get_children()[-2], ax=ax[0])
kwant.plotter.current(syst, current_K, cmap=blue_cm, ax=ax[1], linecolor=cmocean.cm.ice_r(0.75), density=0.5, relwidth=0.1)
plt.colorbar(ax[1].get_children()[-2], ax=ax[1])

patch_up = mpatches.Patch(color=cmocean.cm.amp(0.75), label='$k\'$')
patch_down = mpatches.Patch(color=cmocean.cm.ice_r(0.75), label='$k$')

# plt.legend(handles=[patch_up, patch_down])
# plt.xlabel('x-coordinate [\AA]')
# plt.ylabel(r'y-coordinate [\AA]')
plt.tight_layout()
# plt.savefig('spin-densities.pdf')
plt.show()