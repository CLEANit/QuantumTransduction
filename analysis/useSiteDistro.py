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

ga = dill.load(open('restart/ga.dill', 'rb'))
structures = ga.current_structures
site_distro = dill.load(open('site_distro.dill', 'rb'))

averages = {}
for k, v in site_distro.items():
    print(k, np.mean(v))