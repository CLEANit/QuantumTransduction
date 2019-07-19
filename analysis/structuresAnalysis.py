#!/usr/bin/env python

import dill
import numpy as np
import matplotlib.pyplot as plt
import kwant
import cmocean
import progressbar



font = {'family' : 'CMU Sans Serif',
#         'weight' : 'light',
        'size'   : 12}
plt.rc('font', **font)
plt.rc('text', usetex=True)

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
for energy in bar(energies):
	smatrix = kwant.smatrix(best_structure.system, energy)
	positives = np.where(smatrix.lead_info[0].velocities <= 0)[0]
	momentas = smatrix.lead_info[0].momenta[positives]
	K_prime_indices = np.where(momentas < 0)[0]
	K_indices = np.where(momentas >= 0)[0]

	K_prime_wfs = kwant.wave_function(best_structure.system, energy)(0)[K_prime_indices,:]
	K_wfs = kwant.wave_function(best_structure.system, energy)(0)[K_indices,:]
	currents_kp.append(np.sum(J(wf) for wf in K_prime_wfs))
	currents_k.append(np.sum(J(wf) for wf in K_wfs))

current_K_prime = np.sum(np.array(currents_kp))
current_K = np.sum(np.array(currents_k))

fig, ax = plt.subplots(1,1, figsize=(20, 10))
red_cm = cmocean.cm.amp
red_cm._init()
alphas = np.abs(np.linspace(0.5, 1.0, red_cm.N))
red_cm._lut[:-3,-1] = alphas

blue_cm = cmocean.cm.ice_r
blue_cm._init()
alphas = np.abs(np.linspace(0.25, 1.0, blue_cm.N))
blue_cm._lut[:-3,-1] = alphas

kwant.plot(best_structure.system, 
           site_lw=0.1,
           lead_site_lw=0,
           site_size=0.0,
           site_color=to_rgba('g', 0.1),
           hop_lw=0.1,
           hop_color=to_rgba('g', 0.5),
           lead_color=to_rgba('r', 0.5),
           colorbar=False,
           show=False,
           ax=ax)
kwant.plotter.current(best_structure.system, current_K_prime, cmap=red_cm, ax=ax, linecolor=cmocean.cm.amp(0.75), density=0.5, relwidth=0.1)
kwant.plotter.current(best_structure.system, current_K, cmap=blue_cm, ax=ax, linecolor=cmocean.cm.ice_r(0.75), density=0.5, relwidth=0.1)
patch_up = mpatches.Patch(color=cmocean.cm.amp(0.75), label='$k\'$')
patch_down = mpatches.Patch(color=cmocean.cm.ice_r(0.75), label='$k$')

plt.legend(handles=[patch_up, patch_down])
plt.xlabel('x-coordinate [\AA]')
plt.ylabel(r'y-coordinate [\AA]')
plt.tight_layout()
# plt.savefig('spin-densities.pdf')
plt.savefig('valley_densities.pdf')
plt.show()
# best_structure.visualizeSystem(args={'file': 'best_structure.pdf'})

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