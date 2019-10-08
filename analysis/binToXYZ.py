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
from ase import Atoms

dir_list = subprocess.check_output('ls -d run*', shell=True).split()
bar = progressbar.ProgressBar()
coords = open('coordinates.xyz', 'w')

for dirname in bar(dir_list):
    os.chdir(dirname)
    ga = dill.load(open('restart/ga.dill', 'rb'))
    structures = ga.current_structures

    s1 = structures[0]
    counter = 0
    for site in s1.system_colours:
        if s1.body(site.pos):
            counter += 1
    for s in structures:
        colours = s.system_colours
        print(str(counter) + '\n', file=coords)
        for site in colours:
            if s1.body(site.pos):
                if colours[site]:
                    print('P', site.pos[0], site.pos[1], 0.0, file=coords)
                else:
                    print('N', site.pos[0], site.pos[1], 0.0, file=coords)
    os.chdir('..')

