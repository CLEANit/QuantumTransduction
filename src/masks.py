#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

'''
    The functions here can be thought of as masks that 
    go overtop of the crystal lattice. Once you apply a
    mask, the atoms in certain regions are then removed.

    All of the functions have to recieve the args: maxs, tags.
'''


def sierpinskiMask(maxs, tags):
    '''
        This is a mask related to fractals, called the Sierpinski mask.
    '''

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

def randomBlockHoles(n, r, max_tag, min_pos, max_pos, pos):
    '''
        This places n random squares of size r by r randomly in the lattice.
    '''

    hx = (max_pos[0] - min_pos[0]) / max_tag[0]
    hy = (max_pos[1] - min_pos[1]) / max_tag[1]

    rx = np.random.uniform(min_pos[0] + r, max_pos[0] - r, n)
    ry = np.random.uniform(min_pos[1] + r, max_pos[1] - r, n)

    x = np.linspace(min_pos[0], max_pos[0], max_tag[0] + 1)
    y = np.linspace(min_pos[1], max_pos[1], max_tag[1] + 1)
    img = np.ones((max_tag[0] + 1, max_tag[1] + 1))

    for cx, cy in zip(rx, ry):
        xinds = np.argwhere(np.abs(x - cx) < r )
        yinds = np.argwhere(np.abs(y - cy) < r )
        for xi in xinds:
            for yi in yinds:
                img[xi, yi] = 0

    return img[(pos[:,0] / hx).astype(int), (pos[:,1] / hy).astype(int)]

def randomCircleHoles(n, r, max_tag, min_pos, max_pos, pos):
    '''
        This places n random circles of size r by r randomly in the lattice.
    '''

    hx = (max_pos[0] - min_pos[0]) / max_tag[0]
    hy = (max_pos[1] - min_pos[1]) / max_tag[1]

    rx = np.random.uniform(min_pos[0] + r, max_pos[0] - r, n)
    ry = np.random.uniform(min_pos[1] + r, max_pos[1] - r, n)
    x = np.linspace(min_pos[0], max_pos[0], max_tag[0] + 1)
    y = np.linspace(min_pos[1], max_pos[1], max_tag[1] + 1)
    Y, X = np.meshgrid(y, x)
    img = np.ones((max_tag[0] + 1, max_tag[1] + 1))
    for x, y in zip(rx, ry):
        img[(X - x)**2 + (Y - y)**2 < r**2] = 0
    return img[(pos[:,0] / hx).astype(int), (pos[:,1] / hy).astype(int)]

def randomBlocksAndCirclesHoles(rbs, rcs, max_tag, min_pos, max_pos, pos):
    '''
        This is a function that recieves the partials of the two functions above.
        This allows for circles and holes to be placed in the lattice.
    '''
    return np.logical_and(rbs(max_tag, min_pos, max_pos, pos), rcs(max_tag, min_pos, max_pos, pos))

def image(name, shape, maxs, tag):

    padding = np.subtract(np.add(maxs, 1), shape)

    if padding[1] % 2 == 0:
        x_padl, x_padr = int(padding[1] / 2), int(padding[1] / 2)
    else:
        x_padl, x_padr = int(padding[1] / 2), int(padding[1] / 2) + 1

    if padding[0] % 2 == 0:
        y_padl, y_padr = int(padding[0] / 2), int(padding[0] / 2)
    else:
        y_padl, y_padr = int(padding[0] / 2), int(padding[0] / 2) + 1



    padding = ((x_padl, x_padr), (y_padl, y_padr))

    img = np.rot90(np.pad(np.round(np.divide(misc.imresize(misc.imread(name, mode='L'), (shape[1], shape[0])), 255.)).astype(int), padding, mode='constant', constant_values=1),k=3)
    # print(img.shape, shape, padding, maxs)

    return img[tags[:,0], tags[:,1]]
    