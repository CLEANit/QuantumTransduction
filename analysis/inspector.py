#!/usr/bin/env python


from qmt.system import Structure
from qmt.generator import Generator
from qmt.ga import GA
from qmt.serializer import Serializer
from qmt.parser import Parser
from qmt.timer import Timer
from qmt.parser import Parser

import numpy as np
import os

import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool

import coloredlogs, verboselogs
import copy
import matplotlib.pyplot as plt
import pickle

font = {'family' : 'CMU Serif',
#         'weight' : 'light',
        'size'   : 18}

plt.rc('font', **font)
plt.rc('text', usetex=True)


# create logger
coloredlogs.install(level='INFO')

logger = verboselogs.VerboseLogger('qmt::runner ')

# def threadedCall(structure, lead0, lead1):
#     return structure.getCurrent(lead0, lead1, avg_chem_pot=2.7)

def objectiveFunction():
    pass


def main():
    total_timer = Timer()
    iteration_timer = Timer()
    short_timer = Timer()
    total_timer.start()

    logger.success(' --- Welcome to the Kwantum Transmission Device Optimizer --- ')

    parser = Parser()
    pool = Pool(nodes=parser.config['n_cpus'])
    logger.info('Running calculations with ' + str(parser.config['n_cpus']) + ' workers.')
            
    ga = GA(parser, objective_function=objectiveFunction)
    structures = ga.generator.generateAll(pool=pool, seeds=np.random.randint(0, 2**32 - 1, parser.config['GA']['n_structures']))

    s = structures[0]


    fig, axes = plt.subplots(2, 2, figsize=(10,10))
    
    ms, bs = s.getBandStructure(0)
    axes[0][0].plot(ms, bs, c='k')
    axes[0][0].set_xlabel('Wavenumber [\AA${}^{-1}$]')
    axes[0][0].set_ylabel('Energy [eV]')


    es, cs = s.getConductance(0, 1)
    axes[0][1].plot(es, cs, c='k')
    axes[0][1].set_ylabel('Conductance [$2e^2 / h$]')
    axes[0][1].set_xlabel('Energy [eV]')

    es, ds = s.getDOS()
    axes[1][0].plot(es, ds, c='k')
    axes[1][0].set_ylabel('Arbitrary Units')
    axes[1][0].set_xlabel('Energy [eV]')

    energy_range = s.getEnergyRange()
    energies = np.linspace(energy_range[0], energy_range[1], 128)
    cvs = [s.getValleyPolarizedConductance(energy, 0, 1) for energy in energies]
    cvs = np.array(cvs)
    axes[1][1].plot(energies, cvs[:, 0], 'k', label='$k\'$')
    axes[1][1].plot(energies, cvs[:, 1], 'k--', label='$k$')
    axes[1][1].set_ylabel('Conductance [$2e^2 / h$]')
    axes[1][1].set_xlabel('Energy [eV]')
    axes[1][1].legend()

    plt.show()

    logger.success(' --- Elapsed time: %s ---' % (total_timer.stop()))

if __name__ == '__main__':
    main()
