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

    def generate(self, mask, init=True):
        self.n_generated += 1
        m = copy.copy(dill.loads(self.dill_model))
        if init:
            m.applyMask(mask)
            m.attachLeads()
            m.finalize()
        return m 

    # def __reduce__(self):
    #     return (self.__class__, (self.generator, self.n_generated))

class Model:
    def __init__(   self,
                    logger, 
                    shape,
                    body,
                    potential,
                    lead_shapes,
                    lead_vectors,
                    lead_offsets,
                    lead_potentials,
                    shape_offset=(0., 0.),
                    lattice_type='graphene',
                    lattice_const=1.0,
                    t=1.0
                 ):

        self.logger = logger
        start = time.clock()
        # type of lattice construction
        if lattice_type == 'graphene':
            self.lattice = kwant.lattice.honeycomb(lattice_const, norbs=1)
            self.a, self.b = self.lattice.sublattices

        # define the class parameters
        self.shape = shape
        self.body = body
        self.potential = potential

        assert len(lead_shapes) == len(lead_vectors) == len(lead_offsets)
        
        self.lead_shapes = lead_shapes
        self.lead_vectors = lead_vectors
        self.lead_offsets = lead_offsets
        self.lead_potentials = lead_potentials

        self.shape_offset = shape_offset
        self.lattice_type = lattice_type
        self.lattice_const = lattice_const 
        self.t = t

        self.system = kwant.Builder()

        self.build()

    def build(self):
        self.system[self.lattice.shape(self.shape, self.shape_offset)] = self.potential
        self.hoppings = self.lattice.neighbors()
        self.system[self.hoppings] = -self.t

    def attachLeads(self):
        self.leads = []
        self.symmetries = []
        for lead_shape, lead_vector, lead_offset, lead_pot in zip(self.lead_shapes, self.lead_vectors, self.lead_offsets, self.lead_potentials):
            sym = kwant.TranslationalSymmetry(self.lattice.vec(lead_vector))
            lead = kwant.Builder(sym)
            lead[self.lattice.shape(lead_shape, lead_offset)] = lead_pot
            lead[self.hoppings] = -self.t
            self.leads.append(lead)
            self.symmetries.append(sym)
            self.system.attach_lead(lead)

    def applyMask(self, mask):
        tags = []
        positions = []
        sites = []
        for s, v in self.system.site_value_pairs():
            # if the site is in the body
            if self.body(s.pos):
                tags.append(s.tag)
                positions.append(s.pos)
                sites.append(s)
            # print (s.tag)
        tags = np.array(tags)
        positions = np.array(positions)
        min_tag_sx = np.min(tags[:,0])
        min_tag_sy = np.min(tags[:,1])
        min_pos_sx = np.min(positions[:,0])
        min_pos_sy = np.min(positions[:,1])
        max_pos_sx = np.max(positions[:,0])
        max_pos_sy = np.max(positions[:,1])

        tag_length = np.max(tags[:,0]) - min_tag_sx
        tag_width = np.max(tags[:,1]) - min_tag_sy

        tags[:,0] += np.abs(min_tag_sx)
        tags[:,1] += np.abs(min_tag_sy)
        positions[:, 0] += np.abs(min_pos_sx)
        positions[:, 1] += np.abs(min_pos_sy)

        removed_tags = np.argwhere(mask((tag_length, tag_width), (min_pos_sx, min_pos_sy), (max_pos_sx, max_pos_sy), positions) == 0).astype(int)
        removed_tags = removed_tags.reshape(removed_tags.shape[0]).astype(int)
        for elem in removed_tags:
            del self.system[sites[int(elem)]]
        self.system.eradicate_dangling()

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

    def plotCurrent(self, lead_id, energy=-1, args={}):
        start = time.clock()
        J = kwant.operator.Current(self.system)
        current = np.sum(J(p) for p in self.getWaveFunction(lead_id, energy))
        self.logger.info('Current calculation took %0.2f seconds.' % (time.clock() - start))
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
        cut = lambda site_to, site_from : site_from.pos[0] > val and site_to.pos[0] <= val
        J = kwant.operator.Current(self.system, where=cut, sum=True)
        return J(self.getWaveFunction(0)[0])

    def birth(self, parents, conditions):
        self.system = kwant.Builder()
        n_parents = len(parents)

        for parent, condition in zip(parents, conditions):
            syst = parent.getPreSystem()
            sites = syst.sites()
            sites_to_del = []
            for s in sites:
                if not condition(s.pos):
                    sites_to_del.append(s)
            for elem in sites_to_del:
                del syst[elem]

            self.system.update(syst)
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