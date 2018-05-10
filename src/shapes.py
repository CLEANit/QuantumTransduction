#!/usr/bin/env python

import numpy as np

sin_30, cos_30 = (1 / 2., np.sqrt(3) / 2.)


def square(l, coord):
    x, y = coord
    return -l < x < l and -l < y < l

def rectangle(ls, le, ws, we, coord):
    x, y = coord
    return ls < x < le and ws < y < we

def circle(r, coord):
    x, y = coord
    return x**2 + y**2 < r**2

def ring(r1, r2, coord):
    x, y = coord
    return x**2 + y**2 > r1 and x**2 + y**2 < r2

def ellipse(a, b, r, coord):
    return (x / a)**2  + (y / b)**2 < r**2

def rectDevice(body, lc, ruc, rlc, coord, mask=None):
    if mask is None:
        return body(coord) + lc(coord) + ruc(coord) + rlc(coord)
    else:
        return body(coord) + lc(coord) + ruc(coord) + rlc(coord) + mask(coord)

def sierpinskiCarpetMask(coord):
    x, y = coord
    while( x > 0. or y > 0.):
        if int(x / 5.0) == 1 and int(y / 5.0) == 1:
            return False
        x /= 15.
        y /= 15.
    return True