#!/usr/bin/env python


from src.qmt.system import Structure
from src.qmt.generator import Generator
from src.qmt.parser import Parser
from src.qmt.timer import Timer
import dill
import numpy as np

import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool

import coloredlogs, verboselogs
import copy
import matplotlib.pyplot as plt
# create logger
coloredlogs.install(level='INFO')

logger = verboselogs.VerboseLogger('qmt::autoecoder ')


def main():
    total_timer = Timer()
    short_timer = Timer()
    total_timer.start()
    pool = Pool()

    parser = Parser()
    g = Generator(parser)
    
    short_timer.start()
    try:
        structures = dill.load(open('structures.dill', 'rb'))
    except:
        logger.warning('Could not find structures file. Will start fresh.')
        parsers = g.generateAll()
        structures = pool.map(Structure, parsers)

    logger.success('Structures generated/loaded. Elapsed time: %s' % (short_timer.stop()))

    logger.info('Serializing structures.')
    short_timer.start()
    dill.dump(structures, open('structures.dill', 'wb'))

    logger.success('Structures serialized. Elapsed time: %s' % (total_timer.stop()))
        
if __name__ == '__main__':
    main()
