#!/usr/bin/env python

import numpy as np
import subprocess
import copy
import os
from scipy.interpolate import interp1d, griddata

from .io import IO
from .helper import isParetoEfficient

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
        # self.past_objectives = []
        self.past_vectors = None
        self.current_objectives = []
        self.current_vectors = None

        subprocess.run(['mkdir -p output'], shell=True)

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
        """
        Rank the current structures. We rank based on the euclidean distance
        between the vector and the pareto front.

        Returns
        -------
        A list of structures ordered by their distances to the pareto front.
        
        """
        if self.past_vectors is None:
            data = self.current_vectors
        else:
            data = np.vstack((self.past_vectors, self.current_vectors))

        # if we're working in 2D
        if data.shape[1] == 2:
            pareto_points = np.array(sorted(data[isParetoEfficient(data)], key=lambda x: x[0]))
            x = np.linspace(np.min(pareto_points[:,0]), np.max(pareto_points[:,0]), 512)
            pareto_curve = interp1d(pareto_points[:,0], pareto_points[:,1])(x)
            yd = (pareto_curve[:,None] - data[:,1])**2
            xd = (x[:,None] - data[:,0])**2
            r = np.sqrt(xd + yd)
            objectives = np.min(r, axis=0)
            all_structs = self.past_generation + self.current_structures
            self.current_objectives = np.sort(objectives)[:self.parser.getNStructures()]
            return [all_structs[elem] for elem in np.argsort(objectives)[:self.parser.getNStructures()]]

        # if we're working in 3D
        if data.shape[1] == 3:
            pareto_points = np.array(sorted(data[isParetoEfficient(data)], key=lambda x: x[0]))
            x = np.linspace(np.min(pareto_points[:,0]), np.max(pareto_points[:,0]), 512)
            y = np.linspace(np.min(pareto_points[:,1]), np.max(pareto_points[:,1]), 512)
            X, Y = np.meshgrid(x, y)
            interp = griddata(pareto_points[:,:-1], pareto_points[:,2], (X, Y))

            xd = np.min((x[:,None] - data[:,0])**2, axis=0)
            yd = np.min((y[:,None] - data[:,1])**2, axis=0)
            interp_wout_nan = interp[~np.isnan(interp)]
            zd = np.min((interp_wout_nan.flatten()[:,None] - data[:,2])**2, axis=0)
            objectives = np.sqrt(xd + yd + zd)
            all_structs = self.past_generation + self.current_structures
            self.current_objectives = np.sort(objectives)[:self.parser.getNStructures()]
            return [all_structs[elem] for elem in np.argsort(objectives)[:self.parser.getNStructures()]]
        else:
            logger.error('Error in ranking structures, you seem to be using an objective function that is not 2D nor 3D.')


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
        self.phase_space = open('output/phase_space_gen_' + str(self.generationNumber() - 1) + '.dat', 'w')
        self.phase_space.write('# Generation number: %i\n' % self.generationNumber() - 1)
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
        # self.past_objectives = copy.copy(self.current_objectives)
        self.past_vectors = copy.copy(self.current_vectors)
        self.current_vectors = self.objective_function(*args)
        self.step()
