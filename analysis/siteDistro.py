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
import os
import subprocess

dir_list = subprocess.check_output('find . -name output', shell=True).split()
site_distro = {}
bar = progressbar.ProgressBar()
for dirname in bar(dir_list):
    os.chdir(dirname)
    ga = dill.load(open('restart.dill', 'rb'))
    structures = ga.current_structures

    for s in structures:
        colours = s.sytem_colours
        for site, value in system.site_value_pairs():
            if s.body(site.pos):
                if site not in site_distro:
                    site_distro[site] = []
                site_distro[site].append(colours[site])
    os.chdir('../..')
print(site_distro)
