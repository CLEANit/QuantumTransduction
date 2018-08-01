#!/usr/bin/env python

import numpy as np
from functools import reduce
import operator

def rot(vec, theta):
    """
    This is a function that rotates a vector about a certain angle.

    Parameters
    ----------
    vec : Vector to be rotated.
    theta : Angle to rotate the vector.

    Returns
    -------
    A rotated vector.
    """
    rot_mat = np.array([ [np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)] ])
    return rot_mat.dot(vec)

def orthogVecSlope(vec):
    """
    This function finds a (2D) orthogonal vector, and then returns the slope of that vector.

    Parameters
    ----------
    vec : A vector to which we want to find an othogonal vector to.

    Returns
    -------
    The slope of the orthogonal vector found.
    """

    A = np.array([[vec[0], vec[1]], [-vec[1], vec[0]]])
    trans_vec = np.linalg.solve(A, np.array([0., 1.]))
    if trans_vec[0] == 0.:
        a = 0.
    else:
        a = trans_vec[1] / trans_vec[0]
    return a

def fermi(E, mu, kB_T=0.01):
    """
    The Fermi-Dirac function.

    Parameters
    ----------
    E : Energy value.
    mu : Chemical potential value.
    kb_T : Boltzmann constant times the temperature. Default value is 0.01.
    """
    return 1. / (np.exp((E - mu) / (kB_T)) + 1.)

vectorizedFermi = np.vectorize(fermi)

def getFromDict(dictionary, listed_keys):
    """
    This function finds the value of a nested dictionary where the chain is directed by the listed_keys.

    Parameters
    ----------
    dictionary : Nested dictionary to return the value from.
    listed_keys : Path of keys to be followed to obtain the value.

    Returns
    -------
    The value found from the path of keys.

    """
    return reduce(operator.getitem, listed_keys, dictionary)

def setInDict(dictionary, listed_keys, value):
    """
    Set the value for a key in a nested dictionary.
    
    Parameters
    ----------
    dictionary : Nested dictionary to return the value from.
    listed_keys : Path of keys to be followed to obtain the value.
    value : The value to be set.
    """

    getFromDict(dictionary, listed_keys[:-1])[listed_keys[-1]] = value


def pointInHull(point, hull, tol=1e-12):
    """
    Check if a point is inside of a convex hull.

    Parameters
    ----------
    points : A 2D point that is to be checked.
    hull : A convex hull that is to be checked against.
    tol : A numerical tolerance in the check.

    Returns
    -------
    True if the point is in the hull, False if not.
    """

    return all([np.dot(eq[:-1], point) + eq[-1] <= tol for eq in hull.equations])