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

import coloredlogs, verboselogs
import copy
import matplotlib.pyplot as plt
import pickle


def main():

    parser = Parser()
    structure = Structure(parser, 0, [])
    structure.visualizeSystem()
    plt.show()
    valley_current_1 = structure.getValleyPolarizedCurrent(0, 1)
    valley_current_2 = structure.getValleyPolarizedCurrent(0, 2)
    print('Lead 1 valley currents:', valley_current_1)
    print('Lead 2 valley currents:', valley_current_2)
    print('Total current:', np.sum(valley_current_1 + valley_current_2))
if __name__ == '__main__':
    main()
