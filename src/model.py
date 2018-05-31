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
from functools import partial
from .shapes import *
from .masks import *

import coloredlogs, verboselogs
import copy

# create logger
coloredlogs.install(level='INFO')
logger = verboselogs.VerboseLogger(' <-- QMT: model --> ')

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
                    device,
                    leads,
                    shape_offset=(0., 0.),
                    lattice_type='graphene',
                    lattice_const=1.0,
                 ):

        start = time.clock()
        # type of lattice construction
        if lattice_type == 'graphene':
            self.lattice = kwant.lattice.general([[np.sqrt(3.) * lattice_const, 0],[0, lattice_const]], basis=[(0,lattice_const/2.), (lattice_const / (2*np.sqrt(3.)), 0), (np.sqrt(3)*lattice_const/2, 0), (2*lattice_const/np.sqrt(3), lattice_const/2)], norbs=1)

        # define the class parameters
        self.device = device
        self.leads = leads
        self.body = self.device['body']
        self.index = None
    

        self.shape_offset = shape_offset
        self.lattice_type = lattice_type
        self.lattice_const = lattice_const 

        self.system = kwant.Builder()

        self.build()

    def build(self):
        for shape, offset, hopping, potential in zip(self.device['shapes'], self.device['offsets'], self.device['hoppings'], self.device['potentials']):
            self.system[self.lattice.shape(shape, offset)] = potential
            self.neighbors = self.lattice.neighbors()
            self.system[self.neighbors] = hopping

    def attachLeads(self):
        self.system_leads = []
        for l in self.leads:
            lead_shape = l['shape']
            lead_vector = l['symmetry']
            lead_offset = l['offset']
            lead_pot = l['potential']
            lead_hopping = l['hopping']
            sym = kwant.TranslationalSymmetry(self.lattice.vec(lead_vector))
            lead = kwant.Builder(sym)
            lead[self.lattice.shape(lead_shape, lead_offset)] = lead_pot
            lead[self.neighbors] = lead_hopping
            self.system_leads.append(lead)
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
        masc, info = mask((tag_length, tag_width), (min_pos_sx, min_pos_sy), (max_pos_sx, max_pos_sy), positions)
        self.mask_info = info
        removed_tags = np.argwhere(masc == 0).astype(int)
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

    def getBandStructure(self, leadid, momenta):
        bands = kwant.physics.Bands(self.system_leads[leadid].finalized())
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

    def plotBands(self, momenta, leadid=0):
        kwant.plotter.bands(self.system_leads[leadid].finalized())

    def plotConductance(self, energies, start_lead_id=0, end_lead_id=1):
        conductances = self.getConductance(energies, start_lead_id, end_lead_id)
        plt.figure()
        plt.xlabel("energy [t]")
        plt.ylabel("conductance [$e^2/h$]")
        return plt.plot(energies, conductances)

    def getNSites(self):
        return len(list(self.pre_system.site_value_pairs()))

    def getDOS(self):
        return kwant.kpm.SpectralDensity(self.system)()

    def getCurrentForCut(self, xvals, yvals, energy=-1):
        cut = lambda site_to, site_from : site_from.pos[1] >= yvals[0] and site_to.pos[1] <= yvals[1] and site_from.pos[0] >= xvals[0] and site_to.pos[0] <= xvals[1]
        J = kwant.operator.Current(self.system, where=cut, sum=True)
        wfs = self.getWaveFunction(0, energy=energy)
        if wfs.shape[0] > 0:
            return np.sum(J(wf) for  wf in self.getWaveFunction(0, energy=energy))
        else:
            return 0.

    def birth(self, parents, how, conditions=None):
        self.system = kwant.Builder()
        self.build()
        n_parents = len(parents)
        # print(parents)

        if how == 'stack':
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

        elif how == 'averageNanopores':
            mask_infos = []
            for i, mask_info in enumerate(parents[0].mask_info):
                data = {}
                for parent in parents:
                    for key, val in mask_info.items():
                        if key not in data:
                            data[key] = [parent.mask_info[i][key]]
                        else:
                            data[key].append(parent.mask_info[i][key])
                mask_infos.append(data)
            # print(mask_infos)
            averages = []
            for mask_info in mask_infos:
                average = {}
                for key, val in mask_info.items():
                    # print(np.array(val))
                    average[key] = np.mean(np.array(val))
                averages.append(average)
            print(averages)
            masks = []
            for i in range(len(averages)):
                masks.append(partial(whatMask('circle'), **averages[i]))
            self.applyMask(partial(multiMasks, masks))
            self.attachLeads()
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