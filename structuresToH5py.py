#!/usr/bin/env python


from src.qmt.system import Structure
from src.qmt.generator import Generator
from src.qmt.parser import Parser
from src.qmt.timer import Timer
import dill
import numpy as np

import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool

import coloredlogs, verboselogs
import copy
import matplotlib.pyplot as plt
# create logger
coloredlogs.install(level='INFO')

logger = verboselogs.VerboseLogger('qmt::collector ')

import subprocess
import h5py
import progressbar
import os
bar = progressbar.ProgressBar()

files = subprocess.check_output('find . -name ga.dill', shell=True).split()

all_structures = []
h5_file = h5py.File('binary_images.h5')
h5_file.create_dataset('images', shape=(100000, 128, 128, 1))
h5_file.create_dataset('k_prime_purity', shape=(100000, 1))
h5_file.create_dataset('k_purity', shape=(100000, 1))
h5_file.create_dataset('total_current', shape=(100000, 1))
h5_file.create_dataset('k_prime_currents', shape=(100000, 2))
h5_file.create_dataset('k_currents', shape=(100000, 2))
counter = 0
cwd = os.getcwd()
for f in bar(files):
  dirname = os.path.dirname(os.path.realpath(f))
  os.chdir(dirname)
  os.chdir('..')
  restart = (open('restart/ga.dill', 'rb'))
  d = np.loadtxt('output/currents.dat')
  ga = dill.load(restart)
  structures = ga.getCurrentGeneration()
  for i, s in enumerate(structures):
    kpc = d[i][0], d[i][2]
    kc = d[i][1], d[i][3]
    h5_file['images'][counter] = np.pad(s.getBinaryRepresentation(), ((14,15), (20,20)), mode='constant').reshape(128,128,1)
    h5_file['k_prime_currents'][counter] = kpc
    h5_file['k_currents'][counter] = kc
    h5_file['k_prime_purity'][counter] = kpc[0] / (kpc[0] + kc[0])
    h5_file['k_purity'][counter] = kc[1] / (kc[1] + kpc[1])
    h5_file['total_current'][counter] = kpc[0] + kc[1]
    counter += 1
  restart.close()
  os.chdir(cwd)
