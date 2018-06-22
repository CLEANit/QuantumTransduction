#!/usr/bin/env python


from src.model import Model, Generator
from src.ga import GA
from src.serialize import Serializer
from src.parser import Parser
from src.timer import Timer
from src.builder import parentStructure, applyMask, attachLead
from src.parser import Parser
from functools import partial

import numpy as np
from mpi4py import MPI
import kwant

import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool

import coloredlogs, verboselogs
import copy
import matplotlib.pyplot as plt
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

if __name__ == '__main__':
    main()
