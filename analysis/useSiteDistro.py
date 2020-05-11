#!/usr/bin/env python

import dill
import numpy as np
import matplotlib.pyplot as plt
import kwant
import cmocean
import progressbar
from matplotlib.colors import to_rgba
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.cm as pltcm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import subprocess



ga = dill.load(open('restart/ga.dill', 'rb'))
structures = ga.current_structures
site_distro = dill.load(open('site_distro.dill', 'rb'))

averages = {}
for k, v in site_distro.items():
    averages[k] = np.mean(v)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.concatenate((np.ones(n//2)*0.5, np.linspace(minval, maxval, n//2)))))
        #cmap(np.linspace(minval, maxval, n)))
    return new_cmap

s = structures[0]
s.system_colours = averages
fig, ax = plt.subplots(1,1, figsize=(20, 10))
cmap = truncate_colormap(cmocean.cm.balance_r, 0.5, 0.75)
# cmap = pltcm.get_cmap('bwr')
s.visualizeSystem(cmap=cmap, args={'ax': ax})

sm = pltcm.ScalarMappable(norm=plt.Normalize(0,1), cmap=cmap)
sm._A = []


ax.axis('equal')

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='2.5%', pad=0.05)
fig.colorbar(sm, ax=ax, cax=cax, orientation='vertical')

# plt.colorbar(sm)
# plt.colorbar(im)
plt.tight_layout()

plt.savefig('siteHist.pdf')
