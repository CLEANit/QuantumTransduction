#!/usr/bin/env python

from src.model import Model, Generator
from src.timer import Timer
from src.shapes import *
from src.masks import *
import numpy as np
from matplotlib import pyplot
import kwant
import cmocean

from functools import partial

import multiprocessing
from multiprocessing.pool import ThreadPool as Pool
# from pathos.multiprocessing import ProcessingPool as Pool

import coloredlogs, verboselogs
import copy

# create logger
coloredlogs.install(level='INFO')
logger = verboselogs.VerboseLogger('QMtransport')

pot = 0.1

prim_vecs = [[1., 0.], [0.5, 0.8660254]]
l = 200
w = 100

def potential(site):
    '''
        This function defines the potential on each site. This function is passed into the Model class.
    '''
    (x, y) = site.pos
    d = y * cos_30 + x * sin_30
    return pot * np.tanh(d * 0.5)

def gimmeAModel(index, no_init):
    '''
        This is a function designed to return a finalized (see Kwant) Model class, which would then
        be ready to perform calculations.
    '''

    # sometimes, the models generated throw errors when computing things, hence the while loop
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

        body - main part of the device
        lc - left channel of the device
        rc - right channel of the device
        device - puts the components together to form the full device
        lead_shapers - the different leads that will be placed in the channels

        '''
        body = partial(rectangle, -l, l, -w, w)
        lc = partial(rectangle, -l - w/2, -l + 1, -w/4, w/4)
        rc = partial(rectangle, l - 1, l + w/2, -w/4, w / 4)
        device = partial(rectDeviceTwoChannel, body, lc, rc)
        lead_shapes = [partial(rectangle, -50, 50, -w/4, w/4), partial(rectangle, -50, 50, -w/4, w / 4)]

        # random blocks
        rbs = partial(randomBlockHoles, 5, 5)

        # random circles
        rcs = partial(randomCircleHoles, 5, 5)

        m = Model(  
                    index,
                    logger,
                    device,
                    body, 
                    potential,
                    lead_shapes,
                    [(-1,0), (1,0)], 
                    [(0,0), (0, 0)],
                    [-pot, pot, pot],
                    # mask=partial(randomBlocksAndCirclesHoles, rbs, rcs),
                    mask=partial(image, '/Users/b295319/Desktop/logos.png', (500, 200)),
                    shape_offset=(0, 0),
                    no_init=no_init,
                    lattice_const=1.42
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

    n_initial_parents = 1
    n_cpus = multiprocessing.cpu_count()
    total_timer = Timer()
    total_timer.start()
    short_timer = Timer()

    generator = Generator(gimmeAModel)

    logger.success(' --- Welcome to the Quantum transmission device optimizer --- ')
    logger.info('Number of threads found: %i' % (n_cpus))
    pool = Pool(n_cpus)

    logger.info('Generating initial structures.')
    short_timer.start()
    models = pool.map(generator.generate, [False] * n_initial_parents)
    logger.success('Initial structures were generated successfully. (Elasped time: %s)' % (short_timer.stop()))

    # logger.info('Calculating currents.')
    # short_timer.start()
    # currents = pool.map(getCurrents, zip(models, [l + w/4] * n_initial_parents))
    # logger.success('Current calculations finished. (Elapsed time: %s)' % (short_timer.stop()))
    fig, axes = plt.subplots((1,2), figsize=(20,10))
    models[0].visualizeSystem(args={'ax': axes[0], 'site_color': cmocean.cm.dense(1.0), 'site_edgecolor': cmocean.cm.dense(0.5)})
    models[0].plotCurrent(0, args={'ax': axes[1], 'colorbar': False})
    plt.show()
    # child = generator.generate(True)
    # child.birth(models, [lambda site: site[1] > 0, lambda site: site[1] <= 0])

    # child.visualizeSystem()



    logger.success('Optimization process has completed. (Elapsed time: %s)' % (total_timer.stop()))
if __name__ == '__main__':
    main()