#!/usr/bin/env python

import numpy as np
import subprocess
import copy
import os
from .io import IO

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
        self.past_generation = []
        self.generation_number = 0
        self.objective_function = objective_function
        self.io = IO()
        self.past_objectives = []
        self.past_vectors = []
        self.current_objectives = []
        self.current_vectors = []

        subprocess.run(['mkdir -p output'], shell=True)
        if os.path.isfile('output/phase_space.dat'):
            self.phase_space = open('output/phase_space.dat', 'a')
        else:
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

    def rankGeneration(self):
        all_objs = np.concatenate((self.past_objectives, self.current_objectives))
        all_structs = self.past_generation + self.current_structures
        return [all_structs[elem] for elem in np.argsort(all_objs)[:self.parser.getNStructures()]]


    def setNextGeneration(self, structures):
        """
        Update the current structures in the GA, set the previous structures to past generation.

        Parameters
        ----------
        structures : A list of structures.
        """
        self.past_generation = copy.deepcopy(self.current_structures)
        self.current_structures = structures

    def step(self):
        """
        Increment the generation number by 1.
        """
        self.generation_number += 1

    def generationNumber(self):
        """
        Get the generation number of the GA.

        Returns
        -------
        The generation number
        """
        return self.generation_number

    def writePhaseSpace(self, structures):
        """
        Write out the values of the genes, the vectors associated with the multivariate optimization
        and the scalar merit function. The data is written to "output/phase_space.dat".

        Parameters
        ----------
        structures : The structures that contain the information necessary to write out to the file.
    
        """
        self.phase_space.write('# Generation number: %i\n' % self.generationNumber())
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
        self.phase_space.write('\n')
        self.phase_space.flush()

    def calculate(self, args):
        """
        Calculate the scalar value of the objection function which has been passed into the GA class.
        The objective function should also return a vector of values that corresponds to the 
        multivariate optimization problem. Afterwards increment the generation.

        Parameters
        ----------
        args : Arguments that are passed into your objective function. This should be a tuple.

        """
        self.past_objectives = copy.copy(self.current_objectives)
        self.past_vectors = copy.copy(self.current_vectors)
        self.current_objectives, self.current_vectors = self.objective_function(*args)
        self.step()
