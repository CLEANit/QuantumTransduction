#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as plt
import cmocean as cm

font = {'family' : 'CMU Sans Serif',
#         'weight' : 'light',
        'size'   : 12}

plt.rc('font', **font)
plt.rc('text', usetex=True)


def is_pareto_efficient(costs):
    """
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]>=c, axis=1)  # Remove dominated points
    return is_efficient

n_structures = int(sys.argv[2])

data = np.loadtxt(sys.argv[1])

n_gens = data.shape[0] // n_structures

cf = plt.contourf([[0,0], [0,0]], range(1,n_gens+1), cmap=cm.cm.haline_r)

every_gen = 2

for i in range(n_gens):
    if i % every_gen == 0:
        color = cm.cm.haline_r(i / n_gens)
        # alpha = (i + 1) / float(n_gens + 1)
        # alpha=1
        # pareto_front_indices = np.where(data[i*n_structures:(i+1)*n_structures,2] == 0.)
        gen_data = data[i*n_structures:(i+1)*n_structures,:-1]
        pareto = np.array(sorted(gen_data[is_pareto_efficient(gen_data)], key=lambda x: x[0]))

        # sorted_pf_indices = np.argsort(x_pf_data)
        plt.plot(pareto[:,0], pareto[:,1], '-', color=color)
        plt.plot(data[i*n_structures:(i+1)*n_structures,0], data[i*n_structures:(i+1)*n_structures,1], 'o', color=color, alpha=0.25)

cbar = plt.colorbar(cf)
cbar.ax.get_yaxis().set_ticks(np.linspace(0, 1, 5))
cbar.ax.set_yticklabels(range(0, n_gens + 1, n_gens // 4))
cbar.ax.set_ylabel('Generation number', rotation=270, labelpad=20)
plt.xlabel('Purity of $k\'$ in lead 1')
plt.ylabel('Purity of $k$ in lead 2')
plt.tight_layout()
plt.savefig('purity.pdf', bbox_inches='tight')
plt.show()

