#!/usr/bin/env python


from src.system import Structure, Generator
from src.ga import GA
from src.serialize import Serializer
from src.parser import Parser
from src.timer import Timer
from src.parser import Parser

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
    print(s.getEnergyRange())

    logger.success(' --- Elasped time: %s ---' % (total_timer.stop()))
if __name__ == '__main__':
    main()
