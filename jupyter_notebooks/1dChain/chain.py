#!/usr/bin/env python

import kwant # For plotting
from matplotlib import pyplot
from scipy.sparse.linalg import eigs
import numpy as np

def make_system(a=1, t=1.0, L=128):
    # Start with an empty tight-binding system and a single square lattice. 
    # `a` is the lattice constant (by default set to 1 for simplicity.
    lat = kwant.lattice.chain(a)

    syst = kwant.Builder()

    syst[(lat(x) for x in range(L))] = 4 * t

    syst[lat.neighbors()] = -t
    # syst[lat(0), lat(L - 1)] = -t

    lead = kwant.Builder(kwant.TranslationalSymmetry((-a,)))
    lead[lat(0)] = 4 * t
    lead[lat.neighbors()] = -t
    return lead


def main():
    L = 64
    syst = make_system(L=L)
    # Finalize the system.
    syst = syst.finalized()
    k = np.linspace(-np.pi, np.pi, 128)
    kwant.plotter.bands(syst, momenta=k, show=False)
    pyplot.plot(k, 0.5 - np.cos(k))

    pyplot.show()
if __name__ == '__main__':
    main()