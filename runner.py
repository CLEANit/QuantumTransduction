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

# create logger
coloredlogs.install(level='INFO')

logger = verboselogs.VerboseLogger('qmt::runner ')

# def threadedCall(structure, lead0, lead1):
#     return structure.getCurrent(lead0, lead1, avg_chem_pot=2.7)

def getConductances(structure, lead0, lead1):
    return structure.getValleyPolarizedCurrent(lead0, lead1)

def getNewStructure(parser, identifier):
    return Structure(parser, identifier, [[identifier]])

def objectiveFunction(currents_0_1, currents_0_2):
    vectors = []
    objectives = []
    for v1, v2 in  zip(currents_0_1, currents_0_2):
        objective = []
        objective.append((v1[0]) / (v1[0] + v1[1]) - 1)
        objective.append((v2[1]) / (v2[0] + v2[1]) - 1)
        vectors.append((np.abs((v1[0]) / (v1[0] + v1[1])), np.abs((v2[1]) / (v2[0] + v2[1])), v1[0] + v2[1]))

    data = np.array(vectors).reshape(len(vectors), 3)

    return data

def main():
    total_timer = Timer()
    iteration_timer = Timer()
    short_timer = Timer()
    total_timer.start()

    logger.success(' --- Welcome to the Kwantum Transmission Device Optimizer --- ')

    parser = Parser()
    pool = Pool(nodes=parser.config['n_cpus'])
    logger.info('Running calculations with ' + str(parser.config['n_cpus']) + ' workers.')
    
    serializer = Serializer(parser)
    ga = serializer.deserialize()
    if ga is not None:
        # continue from before
        logger.success('Successfully loaded previous GA. Will continue previous calculation.')
    else:
        logger.info('GA starting from scratch.')
        logger.info('Generating initial structures...')
        short_timer.start()

        
        ga = GA(parser, objective_function=objectiveFunction)
        structures = ga.generator.generateAll(pool=pool, seeds=np.random.randint(0, 2**32 - 1, parser.config['GA']['n_structures']))
        ga.setNextGeneration(structures)
        logger.success('Initial structures generated. Elapsed time: %s' % (short_timer.stop()))

    #########################
    # main loop here
    #########################


    while ga.generationNumber() < parser.getNIterations():

        short_timer.start()
        iteration_timer.start()
        # print info about the upcoming calculation
        ga.summarizeGeneration()

        # get the structures we are going to run calculations on
        structures = ga.getCurrentGeneration()

        # plot the systems and save image to disk

        try:
            os.mkdir('output/gen_' + str(ga.generationNumber()).zfill(3))
        except FileExistsError:
            pass

        for i, s in enumerate(structures):
            s.visualizeSystem(args={'dpi': 600, 'file': 'output/' + 'gen_' + str(ga.generationNumber()).zfill(3) + '/gen_%03i_struct_%03i.png' % (ga.generationNumber(), i)})

        # calculate currents and write them out to disk
        currents_0_1 = pool.map(getConductances, structures, [0] * len(structures), [1] * len(structures))
        currents_0_2 = pool.map(getConductances, structures, [0] * len(structures), [2] * len(structures))
        
        ga.io.writer('output/currents_gen_' + str(ga.generationNumber()).zfill(3) + '.dat', '# Currents (lead1-k\', lead1-k, lead2-k\', lead2-k)\n', header=True)
        for cs1, cs2 in zip(currents_0_1, currents_0_2):
            ga.io.writer('output/currents_gen_' + str(ga.generationNumber()).zfill(3) + '.dat', cs1 + cs2)

        # calculate the objective function
        ga.calculate((currents_0_1, currents_0_2))

        structures = ga.rankGeneration()
        logger.success('Calculations finished. Elapsed time: %s' % (short_timer.stop()))
        # write gene variables and objective function parameters to file
        ga.writePhaseSpace(structures)
        # subset_limit = parser.config['GA']['ann-params']['random-step']['keep-best']
        # structures_subset = structures[:subset_limit]
        # new_structures = []
        # for i in range(len(structures)):
        #     index = np.random.randint(subset_limit)
        #     new_structures.append(g.mutateAllWeights(structures_subset[index]))

        short_timer.start()
        pairs = []
        for i, s1 in enumerate(structures):
            for j in range(parser.getGAParameters()['n_children']):
                if i != j:
                    pairs.append((s1, structures[j]))

        pairs = pairs[:len(structures)]
        
        if parser.getGAParameters()['crossing-fraction'] > 0.:
            structures = ga.generator.crossOverAll(pairs, pool=pool, seeds=np.random.randint(0, 2**32 - 1, len(structures)))
        
        # mutate the current generation
        structures = ga.generator.mutateAll(structures, pool=pool, seeds=np.random.randint(0, 2**32 - 1, len(structures)))

        ga.setNextGeneration(structures)
        logger.success('Structures have been updated. Elapsed time: %s' % (short_timer.stop()))
        # print how long it took and serialize the current GA
        short_timer.start()
        serializer.serialize(ga)
        pickle.dump(ga.history, open('output/history.pkl', 'wb'))
        logger.success('Generation %i completed. Elapsed time: %s' % (ga.generationNumber(), iteration_timer.stop()))

    logger.success(' --- Elapsed time: %s ---' % (total_timer.stop()))

if __name__ == '__main__':
    main()
