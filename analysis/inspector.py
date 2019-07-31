#!/usr/bin/env python


from qmt.system import Structure
from qmt.generator import Generator
from qmt.ga import GA
from qmt.parser import Parser
from qmt.timer import Timer

import numpy as np
import os
from pathos.multiprocessing import ProcessingPool as Pool
import coloredlogs, verboselogs
import matplotlib.pyplot as plt
import progressbar as pb

font = {'family' : 'serif',
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

    logger.success(' --- Welcome to the Kwantum Transmission Device Inspector --- ')

    parser = Parser()
    pool = Pool(nodes=parser.config['n_cpus'])
    logger.info('Running calculations with ' + str(parser.config['n_cpus']) + ' workers.')
            
    ga = GA(parser, objective_function=objectiveFunction)
    structures = ga.generator.generateAll(pool=pool, seeds=np.random.randint(0, 2**32 - 1, parser.config['GA']['n_structures']))

    s = structures[0]


    s.visualizeSystem(args={'dpi': 600, 'file': 'system.png'})
    fig, axes = plt.subplots(3, 2, figsize=(10,15))

    
    ms, bs = s.getBandStructure(0)
    axes[0][0].plot(ms, bs, c='k')
    axes[0][0].set_xlabel('Wavenumber [\AA${}^{-1}$]')
    axes[0][0].set_ylabel('Energy [eV]')


    es, cs = s.getConductance(0, 1)
    axes[0][1].plot(es, cs, c='k')
    axes[0][1].set_ylabel('Transmission Function')
    axes[0][1].set_xlabel('Energy [eV]')
    # axes[0][1].set_xlim([-0.5, 0.5])

    es, ds = s.getDOS()
    axes[1][0].plot(es, ds, c='k')
    axes[1][0].set_ylabel('Density of States')
    axes[1][0].set_xlabel('Energy [eV]')
    # axes[1][0].set_xlim([-0.5, 0.5])
    cvs = [s.getValleyPolarizedConductance(energy, 0, 1) for energy in es]
    cvs = np.array(cvs)
    axes[1][1].plot(es, cvs[:, 0], 'k', label='$k\'$')
    axes[1][1].plot(es, cvs[:, 1], 'k--', label='$k$')
    axes[1][1].set_ylabel('Transmission Function')
    axes[1][1].set_xlabel('Energy [eV]')
    # axes[1][1].set_xlim([-0.5, 0.5])
    axes[1][1].legend()

    biases = np.linspace(0.05, 0.5, 8)
    currents = []
    bar = pb.ProgressBar()
    for bias in bar(biases):
        s.parser.config['System']['bias'] = bias
        currents.append(s.getCurrent(0, 1))

    currents = np.array(currents)
    axes[2][0].plot(biases, currents, 'k')
    axes[2][0].set_xlabel('Bias [V]')
    axes[2][0].set_ylabel('Current [$e / \pi \hbar$]')

    s.visualizeSystem(args={'ax': axes[2][1]})


    for axis in fig.get_axes():
        axis.grid(linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()

    logger.success(' --- Elapsed time: %s ---' % (total_timer.stop()))

if __name__ == '__main__':
    main()
