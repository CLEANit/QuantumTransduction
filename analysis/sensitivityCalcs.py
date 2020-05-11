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
from pathos.multiprocessing import ProcessingPool as Pool

ga = dill.load(open('restart/ga.dill', 'rb'))

structures = ga.rankGenerationWithSquare()

best_structure = structures[0]
print('Got best structure, now calculating currents...')

N = 128
data = open('currents_vs_flips_edges.dat', 'w')

bar = progressbar.ProgressBar()

pool = Pool(40)

def calc(best_structure, pct):
    # best_structure.bitFlips(pct)
    best_structure.bitFlipsNeighbours(pct)
    currents_0_1 = best_structure.getValleyPolarizedCurrent(0, 1)
    currents_0_2 = best_structure.getValleyPolarizedCurrent(0, 2)
    print('Finished with:', pct)
    return currents_0_1[0], currents_0_1[1], currents_0_2[0], currents_0_2[1]

pcts = np.linspace(0, 1, N)
stuff = pool.map(calc, [best_structure] * N, pcts)

for i, elem in enumerate(stuff):
	data.write('%0.20e\t%0.20e\t%0.20e\t%0.20e\t%0.20e\n' % (pcts[i], elem[0], elem[1], elem[2], elem[3]))
data.close()
