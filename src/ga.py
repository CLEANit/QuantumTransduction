#!/usr/bin/env python

import numpy as np
import subprocess

class GA:
    def __init__(self):
        self.models = []
        self.values = []
        self.ranking = []
        self.queue = []
        self.generation = 0
        self.best_model = None
        self.best_value = None
        
        subprocess.call(['mkdir -p output'], shell=True)
        self.best_values = open('output/best_vs_iter.dat', 'w')

    def rank(self, values):
        self.values.append(values)
        self.ranking.append(np.argsort(values))
        self.generation += 1
        if self.best_value == None:
            self.best_value = np.min(values)
            self.best_model = self.models[0][np.argmin(values)]
        else:
            if np.min(values) < self.best_value:
                self.best_value = np.min(values)
                self.best_model = self.models[0][np.argmin(values)]

    def nGenerations(self):
        return self.generation

    def addModelToQueue(self, model):
        self.queue.append(model)

    def getModelsFromQueue(self):
        if self.generation > 1:
            self.models.pop(0)
        self.models.append(self.queue)
        self.queue = []
        return self.models[0]

    def getRandomWeightedParents(self, n):
        weights =  1. / np.array(np.abs(self.values[self.generation - 1]))
        weights /= np.sum(weights)
        
        all_parents = []
        all_parent_indices = []
        for i in range(n):
            keep_going = True
            while keep_going:
                parents = list(np.random.choice(range(len(weights)), size=2, p=weights).astype(int))
                if parents[0] != parents[1] and parents not in all_parent_indices:
                    keep_going = False
            all_parent_indices.append(parents)
            all_parents.append((self.models[0][parents[0]], self.models[0][parents[1]]))
        # print(all_parent_indices)
        return all_parents

    def summarize(self):
        self.best_values.write('%i\t%1.20e\n' % (self.generation - 1, np.min(self.values[self.generation - 1])))
        self.best_values.flush()

    def getAverageSites(self):
        if len(self.queue) == 0:
            return 0.
        total = 0.
        for m in self.queue:
            total += m.getNSites()
        return total / len(self.queue)

    def getBestModel(self):
        return self.best_model

