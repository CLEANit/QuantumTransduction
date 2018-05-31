#!/usr/bin/env python

from src.model import Model, Generator
from src.ga import GA
from src.serialize import Serializer
from src.parser import Parser
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
logger = verboselogs.VerboseLogger(' <-- QMT: runner --> ')

def gimmeAModel(device, leads, shape_offset=(0, 0), lattice_const=1.42):
    '''
        This is a function designed to return a finalized (see Kwant) Model class, which would then
        be ready to perform calculations.
    '''
    m = Model(  
                device, 
                leads,
                shape_offset=shape_offset,
                lattice_const=lattice_const
             )

    return m

def getVerticalCurrents(args):
    model, val = args
    current = model.getCurrentForVerticalCut(val)
    length = current.shape[0] // 2
    if current.shape[0] % 2 == 1:
        return np.sum(current[:length]), np.sum(current[length + 1:])
    else:
        return np.sum(current[:length]), np.sum(current[length:])

def getCurrents(args):
    model, vals = args
    currents =[]
    for val in vals:
        currents.append(model.getCurrentForCut(val[0], val[1]))
    return currents

def main():
    total_timer = Timer()
    total_timer.start()

    logger.success(' --- Welcome to the Quantum transmission device optimizer --- ')

    parser = Parser()
    n_initial_parents = parser.getNStructures()
    n_iterations = parser.getNIterations()
    device = parser.getDevice()
    mask_function = parser.getMaskFunction()
    leads = parser.getLeads()

    pot = -10.0
    pots = np.linspace(-9.9, 1.0, 128)
    current0 = []
    current1 = []
    for p in pots:
        print(p)
        leads[0]['potential'] = pot
        leads[1]['potential'] = p
        leads[2]['potential'] = p
        model = gimmeAModel(device, leads)
        generator = Generator(model)
        m = generator.generate(mask_function)
        # m.visualizeSystem()
        vals0 = []
        vals1 = []
        for e in np.linspace(-10,10,128):
            vals0.append(m.getCurrentForCut((2, 7.5), (2, 6), energy=e))
            vals1.append(m.getCurrentForCut((2, 7.5), (-6, -2), energy=e))
        current0.append(np.sum(vals0))
        current1.append(np.sum(vals1))


    k = np.linspace(np.pi / 1.42, np.pi / 1.42, 128)
    m.plotBands(k)
    energies, spectrum = m.getDOS()
    plt.plot(energies, spectrum)
    plt.show()
    m.plotConductance(np.linspace(-10,10,128), start_lead_id=0, end_lead_id=1)
    plt.show()
    m.plotConductance(np.linspace(-10,10,128), start_lead_id=0, end_lead_id=2)
    plt.show()
    # m.plotCurrent(0)

    # short_timer = Timer()

    # model = gimmeAModel(device, leads)

    # generator = Generator(model)
    # serializer = Serializer()

    # ga = serializer.deserialize()
    # n_threads = multiprocessing.cpu_count()
    # pool = Pool(n_threads)

    # if ga is None:

    #     ga = GA(fresh=True)
    #     logger.info('Generating initial structure.')
    #     short_timer.start()

    #     logger.success('Initial structure was generated successfully. (Elapsed time: %s)' % (short_timer.stop()))

    #     logger.info('Generating %i copies for first generation structures.' % (n_initial_parents))
    #     short_timer.start()

    #     '''
    #         It is actually faster to create the instances in serial for some reason (with 4 cores)...
    #     '''
    #     models = list(map(generator.generate, [mask_function] * n_initial_parents))
    #     # models = pool.map(generator.generate, [partial(randomBlocksAndCirclesHoles, rbs, rcs)] * n_initial_parents)

    #     logger.success('Copies were generated successfully. (Elapsed time: %s)' % (short_timer.stop()))
    #     for m in models:
    #         ga.addModelToQueue(m, n_initial_parents)
    # else:
    #     logger.success('GA saved on disk has been loaded into memory. Will continue with this GA.')


    # ''' 
    # main loop
    # '''

    # # fig, axes = plt.subplots(10,10, figsize=(10,10))
    # for i in range(n_iterations):
    # # for model in models:
    # #     model.visualizeSystem()
    #     short_timer.start()
    #     logger.info('Average number of sites for upcoming model calculations: %0.2f' % (ga.getAverageSites()))
    #     models = ga.getModelsFromQueue()
    #     if len(models) != 0:
    #         # logger.info('Calculating current densities.')
    #         currents = pool.map(getCurrents, zip(models, [[[(14.5, 30.), (14.5, 15.5)], [(14.5, 30.), (-15.5, -14.5)]]] * len(models)))
    #         currents = np.array(currents)
    #         # logger.success('Current density calculations finished. (Elapsed time: %s)' % (short_timer.stop()))
    #         values = []
    #         for cu in currents:
    #             print(cu, np.abs(cu[0] / cu[1] - 1.))
    #             values.append(np.abs(cu[0] / cu[1] - 1.))
    #         # print(values)
    #         # for j in range(10):
    #         #     ga.models[0][j].visualizeSystem(args={'ax': axes[i][j]})
    #         ga.rank(values)

    #         parents = ga.getBest(n_initial_parents)
    #         for i in range(len(parents)):
    #             child = generator.generate(init=False)
    #             child.birth(parents[i], 'averageNanopores')
    #             ga.addModelToQueue(child, n_initial_parents)
    #         ga.summarize()
    #         logger.success('GA generation %i finished. (Elapsed time: %s)' % (ga.nGenerations(), short_timer.stop()))


    #     else:
    #         logger.info('There are no more models in the queue. GA will stop now.')
    #         break
    # # plt.show()

    # ga.best_model.visualizeSystem()
    # ga.best_model.plotCurrent(0)
    # short_timer.start()
    # logger.info('Serializing GA.')
    # serializer.serialize(ga)
    # logger.success('Serialization finished. (Elapsed time: %s)' % (short_timer.stop()))

    # # print(models[0].getNSites())
    # # fig, axes = plt.subplots(1, 1, figsize=(20,10))
    # # models[0].visualizeSystem(args={'ax': axes, 'site_color': cmocean.cm.dense(1.0), 'site_edgecolor': cmocean.cm.dense(0.5)})
    # # axes.axis('equal')
    # # plt.savefig('structure.pdf')

    # # fig = models[0].plotCurrent(0, args={'colorbar': True, 'show': False, 'fig_size': (20, 10)})
    # # plt.axis('equal')
    # # plt.savefig('current.pdf')


    # # child = generator.generate(True)
    # # child.birth(models, [lambda site: site[1] > 0, lambda site: site[1] <= 0])

    # # child.visualizeSystem()



    # logger.success('Optimization process has completed. (Elapsed time: %s)' % (total_timer.stop()))
if __name__ == '__main__':
    main()
