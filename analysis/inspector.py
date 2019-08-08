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
plt.rc('text', usetex=False)


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
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(40,10))
    outer = gridspec.GridSpec(2, 1)

    top = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[0], wspace=0.2, hspace=0.2)

    bs_axis = plt.Subplot(fig, top[0])
    ms, bs = s.getBandStructure(0)
    bs_axis.plot(ms, bs, c='k')
    bs_axis.set_xlabel('Wavenumber [\AA${}^{-1}$]')
    bs_axis.set_ylabel('Energy [eV]')
    fig.add_subplot(bs_axis)

    cs_axis = plt.Subplot(fig, top[1])
    es, cs = s.getConductance(0, 1)
    cs_axis.plot(es, cs, c='k')
    cs_axis.set_ylabel('Transmission Function')
    cs_axis.set_xlabel('Energy [eV]')
    fig.add_subplot(cs_axis)

    dos_axis = plt.Subplot(fig, top[2])
    es, ds = s.getDOS()
    dos_axis.plot(es, ds / np.sum(ds), c='k')
    dos_axis.set_ylabel('Density of States')
    dos_axis.set_xlabel('Energy [eV]')
    dos_axis.set_ylim([0.0, 0.1 * np.max(ds / np.sum(ds))])
    fig.add_subplot(dos_axis)

    vcs_axis = plt.Subplot(fig, top[3])
    cvs = [s.getValleyPolarizedConductance(energy, 0, 1) for energy in es]
    cvs = np.array(cvs)
    vcs_axis.plot(es, cvs[:, 0], 'k', label='$k\'$')
    vcs_axis.plot(es, cvs[:, 1], 'k--', label='$k$')
    vcs_axis.set_ylabel('Transmission Function')
    vcs_axis.set_xlabel('Energy [eV]')
    # vcs_axis.set_xlim([-0.5, 0.5])
    vcs_axis.legend()
    fig.add_subplot(vcs_axis)

    crs_axis = plt.Subplot(fig, top[4])
    biases = np.linspace(0.05, 0.5, 64)
    currents = pool.map(s.getCurrent, [0]*biases.shape[0], [1]*biases.shape[0], biases)
    vcs = pool.map(s.getValleyPolarizedCurrent, [0]*biases.shape[0], [1]*biases.shape[0], biases)
    vcs = np.array(vcs)

    crs_axis.plot(biases, currents, 'k', label='Total')
    crs_axis.plot(biases, vcs[:,0], 'r', label='$k\'$')
    crs_axis.plot(biases, vcs[:,1], 'b', label='$k$')    
    crs_axis.set_xlabel('Bias [V]')
    crs_axis.set_ylabel('Current [$e / \pi \hbar$]')
    crs_axis.legend()
    fig.add_subplot(crs_axis)

    sys_axis = plt.Subplot(fig, outer[1])
    s.visualizeSystem(args={'ax': sys_axis})
    fig.add_subplot(sys_axis)

    for axis in fig.get_axes():
        axis.grid(linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('inspection.pdf')

    plt.show()

    logger.success(' --- Elapsed time: %s ---' % (total_timer.stop()))

if __name__ == '__main__':
    main()
