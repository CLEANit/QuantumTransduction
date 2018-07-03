#!/usr/bin/env python


from src.qmt.system import Structure, Generator
from src.qmt.ga import GA
from src.qmt.serialize import Serializer
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

logger = verboselogs.VerboseLogger(' <-- QMT (runner) --> ')

def main():
    total_timer = Timer()
    total_timer.start()

    logger.success(' --- Welcome to the Quantum transmission device optimizer --- ')

    parser = Parser()
    s = Structure(parser)

    logger.success(' --- Elasped time: %s ---' % (total_timer.stop()))
if __name__ == '__main__':
    main()
