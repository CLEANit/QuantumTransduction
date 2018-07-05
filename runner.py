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

def main():
    total_timer = Timer()
    short_timer = Timer()
    total_timer.start()

    logger.success(' --- Welcome to the Quantum transmission device optimizer --- ')

    serializer = Serializer()
    ga = serializer.deserialize()
    if ga is not None:
        # continue from before
        logger.success('Successfully loaded previous GA. Will continue previous calculation.')
    else:
        logger.info('GA starting from scratch.')
        parser = Parser()
        g = Generator(parser)
        logger.info('Generating initial structures...')
        short_timer.start()
        parsers = g.generateAll()
        structures = [Structure(parser) for parser in parsers]
        logger.success('Initial structures generated. Elapsed time: %s' % (short_timer.stop()))
        
        ga = GA(parser, structures)

    #########################
    # main loop here
    #########################

    # while ga.generationNumber() < parser.getNIterations():
    short_timer.start()
    ga.summarizeGeneration()
    structures = ga.getCurrentGeneration()
    currents_0_1 = [s.getCurrent(0, 1, avg_chem_pot=2.7) for s in structures]
    currents_0_2 = [s.getCurrent(0, 2, avg_chem_pot=2.7) for s in structures]
    logger.info('Calculations finished. Elapsed time: %s' % (short_timer.stop()))

    short_timer.start()
    serializer.serialize(ga)
    logger.success('Serialized the GA successfully. Elapsed time: %s' % (short_timer.stop()))
    logger.success(' --- Elapsed time: %s ---' % (total_timer.stop()))
if __name__ == '__main__':
    main()
