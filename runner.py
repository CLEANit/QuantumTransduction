#!/usr/bin/env python


from src.qmt.system import Structure
from src.qmt.generator import Generator
from src.qmt.ga import GA
from src.qmt.serializer import Serializer
from src.qmt.parser import Parser
from src.qmt.timer import Timer
from src.qmt.parser import Parser

import numpy as np

import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool

import coloredlogs, verboselogs
import copy
import matplotlib.pyplot as plt
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
    short_timer = Timer()
    total_timer.start()
    pool = Pool()

    logger.success(' --- Welcome to the Kwantum transmission device optimizer --- ')

    parser = Parser()
    g = Generator(parser)
    serializer = Serializer(parser)
    ga = serializer.deserialize()
    if ga is not None:
        # continue from before
        logger.success('Successfully loaded previous GA. Will continue previous calculation.')
    else:
        logger.info('GA starting from scratch.')
        logger.info('Generating initial structures...')
        short_timer.start()
        structures = g.generateAll()

        logger.success('Initial structures generated. Elapsed time: %s' % (short_timer.stop()))
        
        ga = GA(parser, structures, objective_function=objectiveFunction)

    #########################
    # main loop here
    #########################


    while ga.generationNumber() < parser.getNIterations():

        short_timer.start()

        # print info about the upcoming calculation
        ga.summarizeGeneration()

        # get the structures we are going to run calculations on
        structures = ga.getCurrentGeneration()

        # plot the systems and save image to disk
        for i, s in enumerate(structures):
            s.visualizeSystem(args={'file': 'output/gen_%03i_struct_%03i.png' % (ga.generationNumber(), i)})

        # calculate currents and write them out to disk
        currents_0_1 = pool.map(getConductances, structures, [0] * len(structures), [1] * len(structures))
        currents_0_2 = pool.map(getConductances, structures, [0] * len(structures), [2] * len(structures))

        ga.io.writer('output/currents_gen_' + str(ga.generationNumber()).zfill(3) + '.dat', '# Currents (lead1-k\', lead1-k, lead2-k\', lead2-k)\n', header=True)
        for cs1, cs2 in zip(currents_0_1, currents_0_2):
            ga.io.writer('output/currents_gen_' + str(ga.generationNumber()).zfill(3) + '.dat', cs1 + cs2)

        # calculate the objective function
        ga.calculate((currents_0_1, currents_0_2))

        structures = ga.rankGeneration()

        # write gene variables and objective function parameters to file
        ga.writePhaseSpace(structures)
        # subset_limit = parser.config['GA']['ann-params']['random-step']['keep-best']
        # structures_subset = structures[:subset_limit]
        # new_structures = []
        # for i in range(len(structures)):
        #     index = np.random.randint(subset_limit)
        #     new_structures.append(g.mutateAllWeights(structures_subset[index]))

        pairs = []
        for i, s1 in enumerate(structures):
            for j, s2 in enumerate(structures):
                if i != j:
                    pairs.append((s1, s2))

        if parser.getGAParameters()['crossing-fraction'] > 0.:
            structures = g.crossOverAll(pairs, pool=pool, seeds=np.random.random_integers(0, 2**32 - 1, len(structures)))
        # mutate the current generation
        structures = g.mutateAll(structures, pool=pool, seeds=np.random.random_integers(0, 2**32 - 1, len(structures)))
        
        ga.setNextGeneration(structures)

        # print how long it took and serialize the current GA
        logger.info('Calculations finished. Elapsed time: %s' % (short_timer.stop()))
        serializer.serialize(ga)
        logger.success('Generation %i completed. Elapsed time: %s' % (ga.generationNumber(), short_timer.stop()))

    logger.success(' --- Elapsed time: %s ---' % (total_timer.stop()))

if __name__ == '__main__':
    main()
