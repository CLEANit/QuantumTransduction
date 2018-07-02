#!/usr/bin/env python

from .helper import rot
import numpy as np


def hexagon(coord, xcoords=None, ycoords=None, shift=[0.0, 0.0]):
    x, y = coord
    m = np.sin(np.pi / 6) / np.cos(np.pi / 6)
    return xcoords[0] <= x <= xcoords[1] and ycoords[0] <= x * m + y <= ycoords[1] and ycoords[0] <= -x * m  + y  <= ycoords[1]

def rectangle(coord, angle=0., xcoords=None, ycoords=None, shift=[0.0, 0.0]):
    x, y = coord
    new_val = rot(coord, angle)
    return xcoords[0] <= new_val[0] - shift[0] <= xcoords[1] and ycoords[0] <= new_val[1] - shift[1] <= ycoords[1]

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
    'ellipse': ellipse,
    'hexagon': hexagon
}

def whatShape(name):
    return shape_dict[name]