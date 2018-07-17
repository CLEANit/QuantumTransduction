#!/usr/bin/env python

import numpy as np
import subprocess
import copy

import coloredlogs, verboselogs
# create logger
coloredlogs.install(level='INFO')
logger = verboselogs.VerboseLogger('qmt::ga ')

class GA:
    """
    Genetic algorithm class. This class holds the structures, creates child configurations based on parents.
    """
    def __init__(self, parser, structures, objective_function=None, fresh=False):
        self.parser = parser
        self.initial_generation = structures
        self.current_structures = structures
        self.past_generation = None
        self.generation_number = 0
        self.objective_function = objective_function

        self.past_objectives = []
        self.past_vectors = []
        self.current_objectives = []
        self.current_vectors = []

        subprocess.run(['mkdir -p output'], shell=True)
        self.phase_space = open('output/phase_space.dat', 'w')

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

    def setNextGeneration(self, structures):
        self.past_generation = copy.deepcopy(self.current_structures)
        self.current_structures = structures

    def step(self):
        self.generation_number += 1

    def generationNumber(self):
        return self.generation_number

    def writePhaseSpace(self, structures):
        for i, s in enumerate(structures):
            c = s.getChromosome()
            for val in c:
                if type(val) == list:
                    self.phase_space.write('%1.20e\t' % (np.mean(val)))
                elif type(val) == float:
                    self.phase_space.write('%1.20e\t' % (val))
            for elem in self.current_vectors[i]:
                self.phase_space.write('%1.20e\t' % elem)
            self.phase_space.write('%1.20e\n' % (self.current_objectives[i]))
        self.phase_space.flush()

    def calculate(self, args):
        self.past_objectives = copy.copy(self.current_objectives)
        self.past_vectors = copy.copy(self.current_vectors)
        self.current_objectives, self.current_vectors = self.objective_function(*args)
        self.step()
