#!/usr/bin/env python

import numpy as np

def rotation(theta):
    return np.array([ [np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)] ])

def rectangle(coord, angle=0., xcoords=None, ycoords=None):
    xshift = xcoords[0] + 0.5 * (xcoords[1] - xcoords[0])
    yshift = ycoords[0] + 0.5 * (ycoords[1] - ycoords[0])
    x, y = coord
    valx = x - xshift
    valy = y -  yshift
    new_val = rotation(angle).dot((valx, valy))
    return xcoords[0] <= new_val[0] + xshift <= xcoords[1] and ycoords[0] <= new_val[1] + yshift <= ycoords[1]

def circle(coord, radius=None):
    x, y = coord
    return x**2 + y**2 <= radius**2

def ring(coord, inner_radius=None, outer_radius=None):
    x, y = coord
    return x**2 + y**2 >= inner_radius and x**2 + y**2 <= outer_radius

def ellipse(coord, a=None, b=None, radius=None):
    return (x / a)**2  + (y / b)**2 <= radius**2

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