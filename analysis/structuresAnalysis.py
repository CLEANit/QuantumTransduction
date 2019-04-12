#!/usr/bin/env python

import dill
import numpy as np
import matplotlib.pyplot as plt
font = {'family' : 'CMU Sans Serif',
#         'weight' : 'light',
        'size'   : 12}
plt.rc('font', **font)
plt.rc('text', usetex=True)

ga = dill.load(open('restart/ga.dill', 'rb'))

structures = ga.getCurrentGeneration()
purities = ga.current_vectors
scores = np.sqrt(purities[:,0]**2 + purities[:,1]**2)
best_score = np.argsort(scores)[0]
print(purities[best_score])
print('Best score:', scores[best_score])
print('Best one can do is 2**0.5 =', 2**0.5)

best_structure = structures[best_score]
best_structure.visualizeSystem()
plt.show()

currents_0_1 = best_structure.getValleyPolarizedCurrent(0, 1)
currents_0_2 = best_structure.getValleyPolarizedCurrent(0, 2)
print(currents_0_1, currents_0_2)
es1, cs1 = best_structure.getConductance(0, 1)
es2, cs2 = best_structure.getConductance(0, 2)

plt.plot(es1, cs1, '-', color=plt.get_cmap('viridis')(0.25))
plt.plot(es2, cs2, '--', color=plt.get_cmap('viridis')(0.75))
plt.legend(['$k\'$', '$k$'])
plt.xlabel('Energy [eV]')
plt.ylabel('Conductance [$2e^2 \hbar^{-1}$]')
plt.show()