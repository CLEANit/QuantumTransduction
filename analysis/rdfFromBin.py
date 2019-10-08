#!/usr/bin/env python

from asap3.analysis.rdf import RadialDistributionFunction
from ase.io import read
import sys
import progressbar
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d as gf
from cmocean import cm

font = {'family' : 'CMU Serif',
#         'weight' : 'light',
        'size'   : 36}

plt.rc('font', **font)
plt.rc('text', usetex=True)

traj = read(sys.argv[1], index='0:200')
traj_og = read(sys.argv[2], index='0:200')

rMax = 15.0
nBins = 1000
x = np.linspace(0, rMax, nBins)
bar = progressbar.ProgressBar()

fig, ax = plt.subplots(1, 1, figsize=(20,10))

RDFobj = None
for atoms in bar(traj):
    atoms.cell = [400, 400, 400]
    if RDFobj is None:
        RDFobj = RadialDistributionFunction(atoms, rMax, nBins)
    else:
        RDFobj.atoms = atoms  # Fool RDFobj to use the new atoms
    RDFobj.update()           # Collect data
rdf_PP = RDFobj.get_rdf(elements=(15, 15))
rdf_PN = RDFobj.get_rdf(elements=(15, 7))
rdf_NN = RDFobj.get_rdf(elements=(7, 7))

bar = progressbar.ProgressBar()
RDFobj = None
for atoms in bar(traj_og):
    atoms.cell = [400, 400, 400]
    if RDFobj is None:
        RDFobj = RadialDistributionFunction(atoms, rMax, nBins)
    else:
        RDFobj.atoms = atoms  # Fool RDFobj to use the new atoms
    RDFobj.update()           # Collect data
rdf_og = RDFobj.get_rdf()
ax.plot(x, gf(rdf_PP + 1, 5) / gf(rdf_og + 1, 5), '-', linewidth=4, color='r', label='p-p doped')
ax.plot(x, gf(rdf_PN + 1, 5) / gf(rdf_og + 1, 5), '--', linewidth=4, color='g', label='p-n doped')
ax.plot(x, gf(rdf_NN + 1, 5) / gf(rdf_og + 1, 5), '-.', linewidth=4, color='b', label='n-n doped')
ax.legend()
ax.grid(linestyle='--', linewidth=0.5)
ax.set_xlabel('Distance [\\AA]')
ax.set_ylabel('$g(r) / g_{CC}(r)$')
plt.tight_layout()
plt.savefig('doping_g_r.pdf')
plt.show()
