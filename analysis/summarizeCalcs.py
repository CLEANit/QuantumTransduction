#!/usr/bin/env python

import numpy as np
import cmocean.cm as cm
import sys
import subprocess
import matplotlib.pyplot as plt
import progressbar as pb
import matplotlib.colors as colors

bar = pb.ProgressBar()

files = subprocess.check_output('find . -name phase*', shell=True)

dir_to_file = {}

for elem in files.split():
    elem = elem.decode('utf-8')
    dir_to_file[elem.split('/')[1]] = elem


dir_to_data = {}
for key, val in dir_to_file.items():
    dir_to_data[key] = np.loadtxt(val)

data_x = []
data_y = []
cs = []

for key, val in dir_to_data.items():
    objs = (val[:,0]**2 + val[:,1]**2 + val[:,2]**2)**0.5
    obj = np.max(objs)
    data_x.append(float(key.split('_')[2]))
    data_y.append(float(key.split('_')[6]))
    cs.append(obj)

plt.scatter(data_x, data_y, c=cs, cmap=cm.deep, s=500)
plt.colorbar()
plt.show()
 
