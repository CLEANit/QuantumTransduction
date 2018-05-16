#!/usr/bin/env python

from src.model import Model, Generator
from src.ga import GA
from src.serialize import Serializer
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
l = 256
w = 128


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


    '''
    Define a three-channel device
    '''
    body = partial(rectangle, -l, l, -w, w)
    lc = partial(rectangle, -l - w/2, -l + 1, -w/4, w/4)
    ruc = partial(rectangle, l - 1, l + w/2, w/4, 3 * w / 4)
    rlc = partial(rectangle, l - 1, l + w/2, -3 * w / 4, -w/4)
    device = partial(rectDeviceThreeChannel, body, lc, ruc, rlc)
    lead_shapes = [partial(rectangle, -50, 50, -w/4, w/4), partial(rectangle, -50, 50, w/4, 3 * w / 4), partial(rectangle, -50, 50, -3 * w / 4, -w/4)]

    '''
    Define a two-channel device

    body - main part of the device
    lc - left channel of the device
    rc - right channel of the device
    device - puts the components together to form the full device
    lead_shapers - the different leads that will be placed in the channels

    '''
    # body = partial(rectangle, -l, l, -w, w)
    # lc = partial(rectangle, -l - w/2, -l + 1, -w/4, w/4)
    # rc = partial(rectangle, l - 1, l + w/2, -w/4, w / 4)
    # device = partial(rectDeviceTwoChannel, body, lc, rc)
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

def main():
    logger.success(' --- Welcome to the Quantum transmission device optimizer --- ')

    n_initial_parents = 10
    total_timer = Timer()
    total_timer.start()
    short_timer = Timer()

    model = gimmeAModel()
    generator = Generator(model)
    serializer = Serializer()

    ga = serializer.deserialize()
    n_threads = multiprocessing.cpu_count()
    pool = Pool(n_threads)

    if ga is None:

        ga = GA()
        logger.info('Generating initial structure.')
        short_timer.start()

        logger.success('Initial structure was generated successfully. (Elapsed time: %s)' % (short_timer.stop()))

        logger.info('Generating %i copies for first generation structures.' % (n_initial_parents))
        # random blocks
        rbs = partial(randomBlockHoles, 1, 5)

        # random circles
        rcs = partial(randomCircleHoles, 1, 5)


        short_timer.start()

        '''
            It is actually faster to create the instances in serial for some reason (with 4 cores)...
        '''
        models = list(map(generator.generate, [partial(randomBlocksAndCirclesHoles, rbs, rcs)] * n_initial_parents))
        # models = pool.map(generator.generate, [partial(randomBlocksAndCirclesHoles, rbs, rcs)] * n_initial_parents)

        logger.success('Copies were generated successfully. (Elapsed time: %s)' % (short_timer.stop()))
        for m in models:
            ga.addModelToQueue(m)
    else:
        logger.success('GA saved on disk has been loaded into memory. Will continue with this GA.')



    ''' 
    main loop
    '''

    generations = 10
    fig, axes = plt.subplots(10,10, figsize=(20,20))
    for i in range(generations):
    # for model in models:
    #     model.visualizeSystem()
        short_timer.start()
        logger.info('Average number of sites for upcoming model calculations: %0.2f' % (ga.getAverageSites()))
        models = ga.getModelsFromQueue()
        if len(models) != 0:
            # logger.info('Calculating current densities.')
            currents = list(map(getCurrents, zip(models, [l + w/4] * len(models))))
            currents = np.array(currents)
            # logger.success('Current density calculations finished. (Elapsed time: %s)' % (short_timer.stop()))
            values = []
            for cu in currents:
                values.append(np.abs((cu[0] / cu[1]) - 1.))
            # print(values)
            for j in range(10):
                ga.models[0][j].visualizeSystem(args={'ax': axes[i][j]})
            ga.rank(values)

            parents = ga.getRandomWeightedParents(n_initial_parents)
            for i in range(len(models)):
                child = generator.generate(init=False)
                child.birth(parents[i], [lambda site: site[1] > 0, lambda site: site[1] <= 0])
                ga.addModelToQueue(child)
            ga.summarize()
            logger.success('GA generation %i finished. (Elapsed time: %s)' % (ga.nGenerations(), short_timer.stop(4)))


        else:
            logger.info('There are no more models in the queue. GA will stop now.')
            break
    plt.show()

    ga.best_model.visualizeSystem()
    ga.best_model.plotCurrent(0)
    short_timer.start()
    logger.info('Serializing GA.')
    serializer.serialize(ga)
    logger.success('Serialization finished. (Elapsed time: %s)' % (short_timer.stop()))

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