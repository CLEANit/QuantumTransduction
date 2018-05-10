#!/usr/bin/env python

import numpy as np
# import matplotlib.pyplot as plt


def sierpinskiMask(maxs, tags):

    img = np.ones((maxs[0] + 1, maxs[1] + 1))
    x = np.zeros(2)
    A = np.array([ [ 0.5, 0] , [0, .5] ])


    b1 = np.zeros(2)

    b2 = np.array([ 0.5, 0])

    b3 = np.array([0.25 , np.sqrt(3) / 4])

    for i in range(2048):
        r = np.fix(np.random.rand() * 3)
        if r == 0:
            x = np.dot(A, x) + b1
        if r == 1:
            x = np.dot(A, x) + b2
        if r == 2:
            x = np.dot(A, x) + b3
        i = int(x[0] * (maxs[0] + 1))
        j = int(x[1] * (maxs[1] + 1))
        img[i][j] = 0
    
    return img[tags[:,0], tags[:,1]]
    