#!/usr/bin/env python

import kwant # For plotting
from matplotlib import pyplot


def make_system(a=1, t=1.0, W=10, L=30):
    # Start with an empty tight-binding system and a single square lattice. 
    # `a` is the lattice constant (by default set to 1 for simplicity.
    lat = kwant.lattice.square(a)

    syst = kwant.Builder()

    syst[(lat(x, y) for x in range(L) for y in range(W))] = 4 * t

    syst[lat.neighbors()] = -t

    lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
    lead[(lat(0, j) for j in range(W))] = 4 * t
    lead[lat.neighbors()] = -t
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    return syst

def plot_conductance(syst, energies): # Compute conductance
    data = []
    for energy in energies:
        smatrix = kwant.smatrix(syst, energy)
        data.append(smatrix.transmission(1, 0))
    pyplot.figure()
    pyplot.plot(energies, data)
    pyplot.xlabel("energy [t]")
    pyplot.ylabel("conductance [$e^2/h$]")
    pyplot.show()

def main():
    syst = make_system()
    # Check that the system looks as intended.
    kwant.plot(syst)
    # Finalize the system.
    syst = syst.finalized()
    # We should see conductance steps.
    plot_conductance(syst, energies=[0.01 * i for i in range(100)])

if __name__ == '__main__':
    main()