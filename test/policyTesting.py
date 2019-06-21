#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


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
		if x_nns[xi] >= 0:
			new_xnns.append(x_nns[xi])
		elif x_nns[xi] <= max_i:
			new_xnns.append(x_nns[xi])
	
	for yi in range(len(y_nns)):
		if y_nns[yi] >= 0:
			new_ynns.append(y_nns[yi])
		elif y_nns[yi] <= max_j:
			new_ynns.append(y_nns[yi])

	nns = []
	for xn in new_xnns:
		for yn in new_ynns:
			nns.append((xn, yn))
	
	return np.array(nns)

def contructANN(input_size, hidden_layers=):



def main():
	N, M = 128, 128
	N_trials = N*M

	img = createImage(N=N, M=M)



	for i in range(N_trials):

		# select a random site
		site_index = (np.random.randint(0, N), np.random.randint(0, M))
		neighbours = getNeighbours(site_index[0], site_index[1], N, M)
		get_ann = 

if __name__ == '__main__':
	main()

