#!/usr/bin/env python

import numpy as np
import subprocess
import copy

class GA:
    def __init__(self, fresh=False):
        self.models = []
        self.values = []
        self.ranking = []
        self.queue = []
        self.generation = 0
        self.best_model = None
        self.best_value = None
        self.fresh = fresh
        
        subprocess.call(['mkdir -p output'], shell=True)
        if fresh:
            self.best_values = open('output/best_vs_iter.dat', 'w')
        else:
            self.best_values = open('output/best_vs_iter.dat', 'a')

    def rank(self, values):
        self.values.append(values)
        self.ranking.append(np.argsort(values))
        self.generation += 1
        if self.best_value == None:
            self.best_value = np.min(values)
            self.best_model = self.queue[np.argmin(values)]
        else:
            if np.min(values) < self.best_value:
                self.best_value = np.min(values)
                self.best_model = self.queue[np.argmin(values)]

    def nGenerations(self):
        return self.generation

    def addModelToQueue(self, model, max_num):
        self.queue.append(model)

        if len(self.queue) >= max_num:
            self.queue.pop(0)


    def getModelsFromQueue(self):
        return self.queue

    def getRandomWeightedParents(self, n):
        weights =  1. / np.array(np.abs(self.values[-1]))
        # weights =  np.exp(- np.array(np.abs(self.values[-1])))
        weights /= np.sum(weights)
        
        all_parents = []
        all_parent_indices = []
        for i in range(n):
            keep_going = True
            while keep_going:
                parents = list(np.random.choice(range(len(weights)), size=2, p=weights).astype(int))
                if parents[0] != parents[1] and parents not in all_parent_indices and list(reversed(parents)) not in all_parent_indices:
                    keep_going = False
            # print(all_parent_indices)
            all_parent_indices.append(parents)
            all_parents.append((self.queue[parents[0]], self.queue[parents[1]]))
        # print(all_parent_indices)
        return all_parents

    def getBest(self, n):
        parents = []
        counter = 0
        for incr, i in enumerate(self.ranking[-1]):
            for j in self.ranking[-1][incr + 1:]:
                print(self.values[-1][i], self.values[-1][j])
                parents.append((self.queue[i], self.queue[j]))
                counter += 1
                if counter  == n:
                    break
            if counter  == n:
                break
        return parents

    def summarize(self):
        self.best_values.write('%i\t%1.20e\t%1.20e\n' % (self.generation - 1, np.min(self.values[self.generation - 1]), np.mean(self.values[self.generation - 1])))
        self.best_values.flush()

    def getAverageSites(self):
        if len(self.queue) == 0:
            return 0.
        total = 0.
        for m in self.queue:
            total += m.getNSites()
        return total / len(self.queue)

    def getBestModel(self):
        return self.queue[self.ranking[-1][0]]

