#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import progressbar as pb

def createImage(N=128, M=128, init='uniform'):
    if init == 'uniform':
        return np.ones((N, M)) * 0.5

    if init == 'yramp':
        img = np.zeros((N, M))
        for i in range(M):
            img[:, i] = float(i) / M
        return img

    if init == 'xramp':
        img = np.zeros((N, M))
        for i in range(N):
            img[i, :] = float(i) / N
        return img

    if init == 'ramp':
        img = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                img[i, j] = float(i) / N + float(j) / M
        return img

    if init == 'random':
        return np.round(np.random.rand(N, M))

def getNeighbours(i, j, max_i, max_j):
    x_nns = [i, i - 1, i + 1]
    y_nns = [j, j - 1, j + 1]

    new_xnns = []
    new_ynns = []
    for xi in range(len(x_nns)):
        if x_nns[xi] >= 0 and x_nns[xi] < max_i:
            new_xnns.append(x_nns[xi])
    for yi in range(len(y_nns)):
        if y_nns[yi] >= 0 and y_nns[yi] < max_j:
            new_ynns.append(y_nns[yi])

    nns = []
    for xn in new_xnns:
        for yn in new_ynns:
            nns.append((xn, yn))
    
    return np.array(nns)

def contructANN(input_size, hidden_layers=[16,16]):
    ann = MLPRegressor(hidden_layer_sizes=[input_size] + hidden_layers + [1], activation='logistic')
    ann._random_state = np.random.RandomState(np.random.randint(2**32))
    ann._initialize(np.empty((1, 1)), [input_size] + hidden_layers + [1])
    ann.out_activation_ = 'logistic'
    return ann

def applyPolicy(img, ann):
    N = img.shape[0]
    M = img.shape[1]
    N_trials = N*M
    bar = pb.ProgressBar()
    for i in bar(range(2*N_trials)):

        # select a random site
        site_index = (np.random.randint(0, N), np.random.randint(0, M))
        neighbours = getNeighbours(site_index[0], site_index[1], N, M)

        input_vector = np.zeros((1, 9))
        vals = img[neighbours[:, 0], neighbours[:, 1]]
        input_vector[0, :vals.shape[0]] += vals

        pred = ann.predict(input_vector)[0]
        img[site_index[0], site_index[1]] = np.round(pred)
    return img

class Organism:
    def __init__(self, ann, update_rate=0.5):
        self.ann = ann
        self.update_rate = update_rate

    def updateWeights(self):
        for layer in range(self.ann.n_layers_):
            total_weights = self.ann.coefs_[layer].shape[0] * self.ann.coefs_[layer].shape[1]
            indices_to_update = np.vstack((np.random.randint(0, self.ann.coefs_[layer].shape[0], size=int(total_weights)), np.random.randint(0, self.ann.coefs_[layer].shape[1], size=int(total_weights)))).T
            self.ann.coefs_[layer][indices_to_update[:,0], indices_to_update[:,1]] += self.update_rate * np.random.uniform(low=-1, high=1, size=indices_to_update.shape[0])



def main():
    N_organisms = 28
    N, M = 128, 128
    img = createImage(N=N, M=M, init='uniform')
    organisms = [Organism(contructANN(9)) for _ in range(N_organisms)]



if __name__ == '__main__':
    main()

