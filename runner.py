#!/usr/bin/env python

from src.model import Model
from src.shapes import *
from src.masks import *
import numpy as np
from matplotlib import pyplot
import kwant

from functools import partial

import multiprocessing
from multiprocessing.pool import ThreadPool as Pool
# from pathos.multiprocessing import ProcessingPool as Pool

import coloredlogs, verboselogs
import time


# create logger
coloredlogs.install(level='DEBUG')
logger = verboselogs.VerboseLogger('QMtransport')

pot = 0.1

prim_vecs = [[1., 0.], [0.5, 0.8660254]]
l = 100
w = 50

def potential(site):
    (x, y) = site.pos
    d = y * cos_30 + x * sin_30
    return pot * np.tanh(d * 0.5)

def gimmeAModel(index):

    generate = True

    while generate:

        '''
        Define a three-channel device
        '''
        # body = partial(rectangle, -l, l, -w, w)
        # lc = partial(rectangle, -l - w/2, -l + 1, -w/4, w/4)
        # ruc = partial(rectangle, l - 1, l + w/2, w/4, 3 * w / 4)
        # rlc = partial(rectangle, l - 1, l + w/2, -3 * w / 4, -w/4)
        # device = partial(rectDeviceThreeChannel, body, lc, ruc, rlc)
        # lead_shapes = [partial(rectangle, -50, 50, -w/4, w/4), partial(rectangle, -50, 50, w/4, 3 * w / 4), partial(rectangle, -50, 50, -3 * w / 4, -w/4)]

        '''
        Define a two-channel device
        '''
        body = partial(rectangle, -l, l, -w, w)
        lc = partial(rectangle, -l - w/2, -l + 1, -w/4, w/4)
        rc = partial(rectangle, l - 1, l + w/2, -w/4, w / 4)
        device = partial(rectDeviceTwoChannel, body, lc, rc)
        lead_shapes = [partial(rectangle, -50, 50, -w/4, w/4), partial(rectangle, -50, 50, -w/4, w / 4)]

        # random blocks
        rbs = partial(randomBlockHoles, 10, 5)

        # random circles
        rcs = partial(randomCircleHoles, 10, 5)

        m = Model(  
                    index,
                    logger,
                    device,
                    potential,
                    lead_shapes,
                    [(-1,0), (1,0)], 
                    [(0,0), (0, 0)],
                    [-pot, pot, pot],
                    mask=partial(randomBlocksAndCirclesHoles, rbs, rcs),
                    shape_offset=(0, 0)
                 )
        try:
            m.finalize()
            generate = False
        except:
            pass
    return m

def getCurrents(args):
    model, val = args
    return model.getCurrentForVerticalCut(l + w/2)

def main():

    n_initial_parents = 4
    n_cpus = multiprocessing.cpu_count()
    total_start = time.clock()

    logger.success(' --- Welcome to the Quantum transmission device optimizer --- ')
    logger.info('Number of threads found: %i' % (n_cpus))
    pool = Pool(n_cpus)

    logger.info('Generating initial structures.')
    start = time.clock()
    models = pool.map(gimmeAModel, range(n_initial_parents))
    logger.success('Initial structures were generated successfully. (Elasped time: %0.2f s)' % (time.clock() - start))
    logger.info('Calculating currents.')
    start = time.clock()
    currents = pool.map(getCurrents, zip(models, [l + w/2] * n_initial_parents))
    logger.success('Current calculations finished. (Elapsed time: %0.2f s)' % (time.clock() - start))

    logger.success('Optimization process has completed. (Elapsed time: %0.2f min)' % ((time.clock() - total_start) / 60.))
if __name__ == '__main__':
    main()