#!/usr/bin/env python

import numpy as np
import subprocess
import copy

import coloredlogs, verboselogs
# create logger
coloredlogs.install(level='INFO')
logger = verboselogs.VerboseLogger('qmt::GA ')

class GA:
    """
    Genetic algorithm class. This class holds the structures, creates child configurations based on parents.
    """
    def __init__(self, parser, structures, fresh=False):
        self.parser = parser
        self.initial_generation = structures
        self.current_structures = structures
        self.past_generation = None
        self.iteration_number = 0

    def summarizeGeneration(self):
        """
        Print out the average and standard deviation of the number of orbitals we are currently investigating.
        """
        n_sites = []
        for s in self.current_structures:
            n_sites.append(s.getNSites())
        logger.info('Average number of orbitals: %0.2f +/- %0.2f' % (np.mean(n_sites), np.std(n_sites)))

    def getCurrentGeneration(self):
        """
        Get the current structures in the GA.

        Returns
        -------
        A list of structures.
        """
        return self.current_structures

    def step(self):
        self.iteration_number += 1

    def generationNumber(self):
        return self.iteration_number
