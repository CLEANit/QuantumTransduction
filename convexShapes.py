#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

N = 10

ranges = [[-64, 64], [-32, 32]]

def get(i):
    a = np.random.rand(i, 2)
    a[:, 0] *= (ranges[0][1] - ranges[0][0])
    a[:, 0] += ranges[0][0]
    a[:, 1] *= (ranges[1][1] - ranges[1][0])
    a[:, 1] += ranges[1][0]
    return a

for i in range(N):
    a = get(100)
    hull = ConvexHull(a)

    plt.scatter(a[:, 0], a[:, 1])
    for simplex in hull.simplices:
        plt.plot(a[simplex, 0], a[simplex, 1], 'k-')

    plt.show()


