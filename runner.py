#!/usr/bin/env python

from src.timer import Timer
from src.builder import parentStructure, applyMask, attachLead
from src.parser import Parser
from functools import partial

import numpy as np
from mpi4py import MPI
import kwant

import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool

import coloredlogs, verboselogs
import copy
import matplotlib.pyplot as plt
# create logger
coloredlogs.install(level='INFO')
logger = verboselogs.VerboseLogger(' QMT - runner ')

pot = 0.1

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_procs = comm.Get_size()


def potential(site):
    '''
        This function defines the potential on each site. This function is passed into the Model class.
    '''
    cos_30 = np.cos( np.pi / 6 )
    sin_30 = np.sin( np.pi / 6 )
    (x, y) = site.pos
    d = y * cos_30 + x * sin_30
    return pot * np.tanh(d * 0.5)


def gimmeAModel():
    '''
        This is a function designed to return a finalized (see Kwant) Model class, which would then
        be ready to perform calculations.
    '''

    lead_shapes = [partial(rectangle, -50, 50, -w/4, w/4), partial(rectangle, -50, 50, w/4, 3 * w / 4), partial(rectangle, -50, 50, -3 * w / 4, -w/4)]

    # lead_shapes = [partial(rectangle, -50, 50, -w/4, w/4), partial(rectangle, -50, 50, -w/4, w / 4)]


    m = Model(  
                logger,
                device,
                body, 
                potential,
                lead_shapes,
                [(-1,0), (1,0), (1,0)], 
                [(0,0), (l + w/2, w/4), (l + w/2, -w/4)],
                [-pot, pot, pot],
                shape_offset=(0, 0),
                lattice_const=1.42
             )

    return m

def getCurrents(args):
    model, val = args
    current = model.getCurrentForVerticalCut(l + w/2)
    length = current.shape[0] // 2
    if current.shape[0] % 2 == 1:
        return np.sum(current[:length]), np.sum(current[length + 1:])
    else:
        return np.sum(current[:length]), np.sum(current[length:])


def generateStructures(body, device, pool, n_structures_here, potential=1.):
    if rank == 0:
        total_structures = [None] * n_procs
    else:
        total_structures = None

    total_structures = comm.scatter(total_structures, root=0)

    total_structures  = pool.map(parentStructure, zip([body] * n_structures_here, [device] * n_structures_here, [potential] * n_structures_here, [1.0] * n_structures_here, [1] * n_structures_here))
    return total_structures


def applyMasks(structures, body, maskFunction, pool, n_structures_per_node):
    structures = map(applyMask, zip(structures, [body] * n_structures_per_node, [maskFunction] * n_structures_per_node))
    return structures

def attachLeads(structures, leads, pool, n_structures_per_node):
    structures = pool.map(attachLead, zip(structures, [leads] * n_structures_per_node))
    return structures

def main():

    if rank == 0:
        total_timer = Timer()
        total_timer.start(MPI.Wtime())
        short_timer = Timer()

    if rank == 0:
        logger.info('We found %i MPI processes.' % (n_procs))

    n_threads_per_node = multiprocessing.cpu_count()
    pool = Pool(n_threads_per_node)

    logger.info('Node %i found %i threads to play with.' % (rank, n_threads_per_node))

    if rank == 0:
        parser = Parser()
    else:
        parser = None

    parser = comm.bcast(parser, root=0)

    body, device = parser.getDevice()
    mask_function = parser.getMaskFunction()
    leads = parser.getLeads()
    total_structures = parser.getNStructures()


    n_structures_per_node =  total_structures // n_procs
    if rank == 0:
        logger.info('Each MPI process will handle %i structures.' % (n_structures_per_node))
    
    if rank == 0:
        short_timer.start(MPI.Wtime())   
        logger.info('Generating the initial structures.')

    structures = generateStructures(body, device, pool, n_structures_per_node)
    structures = applyMasks(structures, body, mask_function, pool, n_structures_per_node)
    structures = attachLeads(structures, leads, pool, n_structures_per_node)
    all_structures = comm.gather(structures, root=0)

    if rank == 0:
        # make sure we construct any left over models
        if (total_structures % n_procs) != 0:
            total_structures.append(pool.map(parentStructure, zip([body] * (total_structures % n_procs), [device] * (total_structures % n_procs), [potential] * (total_structures % n_procs), [1.0] * (total_structures % n_procs), [1] * (total_structures % n_procs))))
            total_structures[-1] = map(applyMask, zip(structures, [body] * (total_structures % n_procs), [mask_function] * (total_structures % n_procs)))
            total_structures[-1] = pool.map(attachLead, zip(structures, [leads] * (total_structures % n_procs)))
        all_structures_flatten = [structure for group in all_structures for structure in group]

        for struct in all_structures_flatten:
            kwant.plot(struct)
            plt.show()
        logger.success('Generation of initial structures complete. (Time elapsed: %s)' % (short_timer.stop(MPI.Wtime())))


    if rank == 0:
        logger.success('Optimization process has completed. (Elapsed time: %s)' % (total_timer.stop(MPI.Wtime())))
if __name__ == '__main__':
    main()