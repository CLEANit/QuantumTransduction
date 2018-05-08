#!/usr/bin/env python

from src.model import Model
import numpy as np
from matplotlib import pyplot

sin_30, cos_30 = (1 / 2, np.sqrt(3) / 2)
r = 25.
pot = 0.1

def shape(shift):
    x, y = shift
    return x**2 + y**2 > (r / 2.)**2 and x**2 + y**2 < r**2

def potential(site):
    (x, y) = site.pos
    d = y * cos_30 + x * sin_30
    return pot * np.tanh(d * 0.5)

def lead0_shape(pos):
    x, y = pos
    return -0.4 * r < y < 0.4 * r

def lead1_shape(pos):
    v = pos[1] * sin_30 - pos[0] * cos_30
    return -0.4 * r < v < 0.4 * r

def main():
    lead_shapes = [lead0_shape, lead1_shape]
    m = Model(  shape,
                potential,
                lead_shapes,
                [(-1, 0), (0, 1)], [(0., 0.), (0., 0.)],
                [-pot, pot],
                shape_offset=(r / 2, r / 2)
              )

    syst_fig = m.visualizeSystem()
    pyplot.show()

    m.finalize()
    syst = m.getSystem()
    
    # print m.diagonalize()[0]

    # Compute the band structure of lead 0.
    momenta = [-np.pi + 0.02 * np.pi * i for i in range(101)]
    energies = m.getBandStructure(syst.leads[0], momenta) 
    
    pyplot.figure()
    pyplot.plot(momenta, energies)
    pyplot.xlabel("momentum [(lattice constant)^-1]")
    pyplot.ylabel("energy [t]")
    pyplot.show()

    # Plot conductance.
    energies = [-2 * pot + 4. / 50. * pot * i for i in range(51)]
    conductances = m.getConductance(energies, 0, 1)
    
    pyplot.plot(energies, conductances)
    pyplot.xlabel("energy [t]")
    pyplot.ylabel("conductance [$e^2/h$]")
    pyplot.show()

if __name__ == '__main__':
    main()