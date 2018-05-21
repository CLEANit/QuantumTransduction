#!/usr/bin/env python

import numpy as np

sin_30, cos_30 = (1 / 2., np.sqrt(3) / 2.)


def rectangle(coord, xcoords=None, ycoords=None):
    x, y = coord
    return xcoords[0] <= x <= xcoords[1] and ycoords[0] <= y <= ycoords[1]

def circle(r, coord):
    x, y = coord
    return x**2 + y**2 < r**2

def ring(r1, r2, coord):
    x, y = coord
    return x**2 + y**2 > r1 and x**2 + y**2 < r2

def ellipse(a, b, r, coord):
    return (x / a)**2  + (y / b)**2 < r**2

def nBodyDevice(components, coord):
    firstComp = components[0](coord)
    for i in range(1, len(components)):
        firstComp += components[i](coord)
    return firstComp

shape_dict = {
    'rectangle': rectangle,
    'circle': circle,
    'ring': ring,
    'ellipse': ellipse
}

def whatShape(name):
    return shape_dict[name]