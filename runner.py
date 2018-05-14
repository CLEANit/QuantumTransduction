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
from pathos.multiprocessing import ProcessingPool as Pool

import coloredlogs, verboselogs
import copy

# create logger
coloredlogs.install(level='INFO')
logger = verboselogs.VerboseLogger('QMtransport')

pot = 0.1

prim_vecs = [[1., 0.], [0.5, 0.8660254]]
l = 128
w = 64

def potential(site):
    '''
        This function defines the potential on each site. This function is passed into the Model class.
    '''
    (x, y) = site.pos
    d = y * cos_30 + x * sin_30
    return pot * np.tanh(d * 0.5)


def gimmeAModel():
    '''
        This is a function designed to return a finalized (see Kwant) Model class, which would then
        be ready to perform calculations.
    '''


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


    m = Model(  
                logger,
                device,
                body, 
                potential,
                lead_shapes,
                [(-1,0), (1,0)], 
                [(0,0), (0, 0)],
                [-pot, pot, pot],
                # mask=partial(randomBlocksAndCirclesHoles, rbs, rcs),
                # mask=partial(image, '/Users/b295319/Desktop/logos.png', (400, 200)),
                shape_offset=(0, 0),
                lattice_const=1.42
             )

    return m

def getCurrents(args):
    model, val = args
    return model.getCurrentForVerticalCut(l + w/2)

def main():

    n_initial_parents = 64
    total_timer = Timer()
    total_timer.start()
    short_timer = Timer()

    logger.success(' --- Welcome to the Quantum transmission device optimizer --- ')

    logger.info('Generating initial structure.')
    short_timer.start()
    model = gimmeAModel()
    generator = Generator(model)
    logger.success('Initial structure was generated successfully. (Elapsed time: %s)' % (short_timer.stop()))

    logger.info('Generating %i copies for first generation structures.' % (n_initial_parents))
    # random blocks
    rbs = partial(randomBlockHoles, 1, 5)

    # random circles
    rcs = partial(randomCircleHoles, 1, 5)


    short_timer.start()
    pool = Pool(multiprocessing.cpu_count())

    '''
        It is actually faster to create the instances in serial for some reason (with 4 cores)...
    '''
    models = list(map(generator.generate, [partial(randomBlocksAndCirclesHoles, rbs, rcs)] * n_initial_parents))
    # models = pool.map(generator.generate, [partial(randomBlocksAndCirclesHoles, rbs, rcs)] * n_initial_parents)

    logger.success('Copies were generated successfully. (Elapsed time: %s)' % (short_timer.stop()))
    

    ''' 
    main loop
    '''





    # for model in models:
    #     model.visualizeSystem()
    logger.info('Calculating current densities.')
    short_timer.start()
    currents = pool.map(getCurrents, zip(models, [l + w/4] * n_initial_parents))
    logger.success('Current density calculations finished. (Elapsed time: %s)' % (short_timer.stop()))
    # print(models[0].getNSites())
    # fig, axes = plt.subplots(1, 1, figsize=(20,10))
    # models[0].visualizeSystem(args={'ax': axes, 'site_color': cmocean.cm.dense(1.0), 'site_edgecolor': cmocean.cm.dense(0.5)})
    # axes.axis('equal')
    # plt.savefig('structure.pdf')

    # fig = models[0].plotCurrent(0, args={'colorbar': True, 'show': False, 'fig_size': (20, 10)})
    # plt.axis('equal')
    # plt.savefig('current.pdf')


    # child = generator.generate(True)
    # child.birth(models, [lambda site: site[1] > 0, lambda site: site[1] <= 0])

    # child.visualizeSystem()



    logger.success('Optimization process has completed. (Elapsed time: %s)' % (total_timer.stop()))
if __name__ == '__main__':
    main()