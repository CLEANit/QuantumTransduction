#!/usr/bin/env python

import kwant
import scipy.sparse.linalg as sla
import cmocean
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import multiprocessing
import dill

class Generator:
    def __init__(self, model):
        self.model = model
        self.dill_model = dill.dumps(model)
        self.n_generated = 0

    def generate(self, mask=None, init=True):
        self.n_generated += 1
        m = copy.copy(dill.loads(self.dill_model))
        m.index = self.n_generated
        if init:
            if mask is not None:
                m.applyMask(mask)
            m.attachLeads()
            m.finalize()
        return m 

    # def __reduce__(self):
    #     return (self.__class__, (self.generator, self.n_generated))

class Model:
    def __init__(   self,
                    potential,
                    lead_shapes,
                    t=1.0
                 ):

        start = time.clock()
        
        # define the class parameters
        self.potential = potential
        self.index = None        
        self.t = t

        self.build()

    def build(self):
        self.system[self.lattice.shape(self.shape, self.shape_offset)] = self.potential
        self.hoppings = self.lattice.neighbors()
        self.system[self.hoppings] = -self.t

    def attachLeads(self, lead_shapes, lead_vectors, lead_offsets, lead_potentials):
        self.leads = []
        self.symmetries = []
        for lead_shape, lead_vector, lead_offset, lead_pot in zip(lead_shapes, lead_vectors, lead_offsets, lead_potentials):
            sym = kwant.TranslationalSymmetry(self.lattice.vec(lead_vector))
            lead = kwant.Builder(sym)
            lead[self.lattice.shape(lead_shape, lead_offset)] = lead_pot
            lead[self.hoppings] = -self.t
            self.leads.append(lead)
            self.symmetries.append(sym)
            self.system.attach_lead(lead)

    def finalize(self):
        self.pre_system = self.system
        self.system = self.system.finalized()

    def getPreSystem(self):
        return self.pre_system

    def getSystem(self):
        return self.system

    def getLeads(self):
        return self.leads

    def getPrimVecs(self):
        return self.lattice.prim_vecs

    def family_colors(self, site):
        return 0 if site.family == self.a else 1

    def visualizeSystem(self, args={}):
        return kwant.plot(self.system, site_lw=0.1, colorbar=False, **args)

    '''
    TODO: pass args to the eigensolver
    '''
    def diagonalize(self):
        # Compute some eigenvalues of the closed system
        sparse_mat = self.system.hamiltonian_submatrix(sparse=True)

        return sla.eigs(sparse_mat)

    def getConductance(self, energies, start_lead_id, end_lead_id):
        # Compute transmission as a function of energy
        data = []
        for energy in energies:
            smatrix = kwant.smatrix(self.system, energy)
            data.append(smatrix.transmission(start_lead_id, end_lead_id))
        return data

    def getBandStructure(self, lead, momenta):
        bands = kwant.physics.Bands(lead)
        energies = [bands(k) for k in momenta]
        return energies

    def getWaveFunction(self, lead_id, energy=-1):
        return kwant.wave_function(self.system, energy)(lead_id)

    # currently not working
    def plotWaveFunction(self, lead_id, energy=0., cmap=cmocean.cm.dense):
        return kwant.plotter.map(self.system, np.absolute(self.getWaveFunction(lead_id, energy)[0])**2, oversampling=10, cmap=cmap)

    def plotCurrent(self, lead_id, energy=0., args={}):
        J = kwant.operator.Current(self.system)
        current = np.sum(J(p) for p in self.getWaveFunction(lead_id, energy))
        return kwant.plotter.current(self.system, current, cmap=cmocean.cm.dense, **args)

    def plotBands(self, momenta, lead_id=0):
        energies = self.getBandStructure(self.system.leads[lead_id], momenta) 
        plt.figure()
        plt.xlabel("momentum [(lattice constant)^-1]")
        plt.ylabel("energy [t]")
        return plt.plot(momenta, energies)

    def plotConductance(self, energies, start_lead_id=0, end_lead_id=1):
        conductances = self.getConductance(energies, 0, 1)
        plt.figure()
        plt.xlabel("energy [t]")
        plt.ylabel("conductance [$e^2/h$]")
        return plt.plot(energies, conductances)

    def getNSites(self):
        return len(list(self.pre_system.site_value_pairs()))

    def getCurrentForVerticalCut(self, val):
        cut = lambda site_to, site_from : site_from.pos[0] >= val and site_to.pos[0] < val 
        J = kwant.operator.Current(self.system, where=cut)
        return J(self.getWaveFunction(0)[0])

    def birth(self, parents, conditions):
        self.system = kwant.Builder()
        n_parents = len(parents)
        # print(parents)
        for parent, condition in zip(parents, conditions):
            syst = copy.deepcopy(parent.getPreSystem())
            sites = syst.sites()
            sites_to_del = []
            for s in sites:
                if not condition(s.pos):
                    sites_to_del.append(s)
            for elem in sites_to_del:
                del syst[elem]
            # print (sites_to_del)
            # for s in sites_to_add:
            self.system.update(syst)
            # self.visualizeSystem()
            # self.attachLeads()
        self.finalize()

    # def __reduce__(self):
    #     return (self.__class__, (
    #                 self.index,
    #                 self.logger,
    #                 self.shape,
    #                 self.body,
    #                 self.potential,
    #                 self.lead_shapes,
    #                 self.lead_vectors,
    #                 self.lead_offsets,
    #                 self.lead_potentials,
    #                 self.mask,
    #                 self.shape_offset,
    #                 self.lattice_type,
    #                 self.lattice_const,
    #                 self.t,
    #                 )
    #             )