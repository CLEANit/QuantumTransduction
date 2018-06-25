#!/usr/bin/env python

import kwant
import matplotlib.pyplot as plt

system = kwant.Builder()

# lattice constant
a = 1

# define the lattice
lattice = kwant.lattice.square(a)

# the coupling parameter
t = 1.

# width and length of crystal
W = 10
L = 30

for i in range(L):
    for j in range(W):
        # on-site part of Hamiltonian
        system[lattice(i, j)] = 4 * t

        # hopping in y direction
        if j > 0:
            system[lattice(i, j), lattice(i, j - 1)] = -t

        # hopping in x direction
        if i > 0:
            system[lattice(i, j), lattice(i-1, j)] = -t

sym_left_lead = kwant.TranslationalSymmetry((-a,0))
left_lead = kwant.Builder(sym_left_lead)

for j in range(W):
    left_lead[lattice(0, j)] = 4 * t
    if j > 0:
        left_lead[lattice(0, j), lattice(0, j - 1)] = -t
    left_lead[lattice(1, j), lattice(0, j)] = -t

system.attach_lead(left_lead)


sym_right_lead = kwant.TranslationalSymmetry((a,0))
right_lead = kwant.Builder(sym_right_lead)

for j in range(W):
    right_lead[lattice(0, j)] = 4 * t
    if j > 0:
        right_lead[lattice(0, j), lattice(0, j - 1)] = -t
    right_lead[lattice(1, j), lattice(0, j)] = -t

system.attach_lead(right_lead)


# Visualize the system
kwant.plot(system)

system = system.finalized()

energies = []
data = []

for ie in range(100):
    energy = ie * 0.01
    smatrix = kwant.smatrix(system, energy)

    energies.append(energy)
    data.append(smatrix.transmission(1, 0))

plt.plot(energies, data)
plt.xlabel('energy [t]')
plt.ylabel('conductance [$e^2/h$]')
plt.show()
