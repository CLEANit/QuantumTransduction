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
best_structure.visualizeSystem()
plt.show()

plt.imshow(np.rot90(best_structure.getBinaryRepresentation(best_structure.pre_system, policyMask=True)))
plt.show()
# best_structure.bitFlipsNeighbours(pct)
