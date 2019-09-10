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


file_list = subprocess.check_output('find . -name "ga.dill"', shell=True).split()
site_distro = {}
bar = progressbar.ProgressBar()
for filename in bar(file_list):
	ga = dill.load(open(filename.decode('utf-8'), 'rb'))
	structures = ga.rankGenerationWithSquare()

	for s in structures:
		colours = s.sytem_colours
		for site, value in system.site_value_pairs():
            if s.body(site.pos):
            	if site not in site_distro:
            		site_distro[site] = []
            	site_distro[site].append(colours[site])
print(site_distro)
