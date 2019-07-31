#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

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

N = 8192
energies = np.linspace(-10, 10, N)
kb_T = 0.025851991

for bias in np.linspace(0.05, 0.5, 16):
	mu_left = bias / 2.0 
	mu_right = -bias / 2.0 
	diff_fermi = vectorizedFermi(energies, mu_left, kb_T) - vectorizedFermi(energies, mu_right, kb_T)
	plt.plot(energies, diff_fermi, label=str(bias))
plt.legend()
plt.show()