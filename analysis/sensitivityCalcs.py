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

ga = dill.load(open('restart/ga.dill', 'rb'))

structures = ga.rankGenerationWithSquare()

best_structure = structures[0]
print('Got best structure, now calculating currents...')

N = 128
data = open('currents_vs_flips.dat', 'w')

bar = progressbar.ProgressBar()
for pct in bar(np.linspace(0, 1, N)):
	best_structure.bitFlip(pct)

	currents_0_1 = best_structure.getValleyPolarizedCurrent(0, 1)
	currents_0_2 = best_structure.getValleyPolarizedCurrent(0, 2)
	data.write('%1.20e\t%1.20e\t%1.20e\t%1.20e\t%1.20e\n' % (pct, currents_0_1[0], currents_0_1[1], currents_0_2[0], currents_0_2[1]))
data.close()
