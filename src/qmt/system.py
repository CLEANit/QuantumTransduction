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

from .helper import *

import coloredlogs, verboselogs
import copy

# create logger
coloredlogs.install(level='INFO')
logger = verboselogs.VerboseLogger(' <-- QMT: model --> ')

class Generator:
    """
    this is a test
    """
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

def hoppingFunction(self, t, phi, site1, site2):
    """
    This is a function which returns the hopping value when a magnetic field is being applied.

    Parameters
    ----------
    t : The hopping parameter without a magnetic field being applied.
    phi : The magnetic flux though a single unit cell.
    site1 : A kwant site for the first lattice site.
    site2 : A kwant site for the second lattice site.

    Returns
    -------
    The hopping parameter with a magnetic field being applied.
    """
    lattice_vectors = self.parser.getLatticeVectors()
    B = phi / (lattice_vectors[0][0] * lattice_vectors[1][1] - lattice_vectors[0][1] * lattice_vectors[1][0])
    p1 = site1.pos
    p2 = site2.pos
    return t * np.exp(1j * np.pi * (p2[0] - p1[0]) * (p2[1] + p1[1]) * B)

def onSiteFunction(self, pot, spin, phi, site):
    """
    This is a function which returns the on-site potential when a magnetic field is being applied.

    Parameters
    ----------
    t : The hopping parameter without a magnetic field being applied.
    phi : The magnetic flux though a single unit cell.
    site1 : A kwant site for the first lattice site.
    site2 : A kwant site for the second lattice site.

    Returns
    -------
    The hopping parameter with a magnetic field being applied.
    """
    lattice_vectors = self.parser.getLatticeVectors()

    B = phi / (lattice_vectors[0][0] * lattice_vectors[1][1] - lattice_vectors[0][1] * lattice_vectors[1][0])
    return pot + spin * B

class Structure:
    """
    This is the main class which stores all of the information needed for kwant as well as functions to make it easier to use kwant (not that it is too difficult).

    Parameters
    ----------
    parser : This takes the YAML configuration file which is read in as a dictionary.

    """
    def __init__(   self,
                    parser
                 ):
        # define the class parameters
        self.parser = parser
        self.device = parser.getDevice()
        self.leads = parser.getLeads()
        self.body = self.device['body']
        self.spin_dep = parser.spinDependent()

        self.build()
        self.attachLeads()

    def build(self):
        # first construct the lattice
        lattice_type = self.parser.getLatticeType()
        lattice_constant =  self.parser.getLatticeConstant()
        lattice_vectors = self.parser.getLatticeVectors()
        lattice_basis = self.parser.getLatticeBasis()
        norbs = self.parser.getNumOrbitals()

        if lattice_type == 'general':
            self.lattice = kwant.lattice.general(lattice_vectors, lattice_basis, norbs=norbs)
            if self.spin_dep:
                self.system_up = kwant.Builder()
                self.system_down = kwant.Builder()
            else:
                self.system = kwant.Builder()

        else:
            logger.error('Sorry, we do not support the lattice type: %s' % (lattice_type))
            exit(-1)


        for shape, offset, hopping, potential, phi in zip(self.device['shapes'], self.device['offsets'], self.device['hoppings'], self.device['potentials'], self.device['phis']):
            
            # if we want to consider spin dependent transport
            if self.spin_dep:
                self.system_up[self.lattice.shape(shape, offset)] = partial(onSiteFunction, self, potential, 1/2, phi)
                self.system_down[self.lattice.shape(shape, offset)] = partial(onSiteFunction, self, potential, -1/2, phi)
                
                self.neighbors = self.lattice.neighbors()
                
                self.system_up[self.neighbors] = partial(hoppingFunction, self, hopping, phi)
                self.system_down[self.neighbors] = partial(hoppingFunction, self, hopping, phi)

            else:
                self.system[self.lattice.shape(shape, offset)] = potential
                self.neighbors = self.lattice.neighbors()
                self.system[self.neighbors] = partial(hoppingFunction, self, hopping, phi)

        if self.spin_dep:
            self.system_up.eradicate_dangling()
            self.system_down.eradicate_dangling()
        else:
            self.system.eradicate_dangling()

    def attachLeads(self):
        if self.spin_dep:
            for l in self.leads:
                lead_range = l['range']
                lead_vector = l['symmetry']
                lead_offset = l['offset']
                lead_pot = l['potential']
                lead_hopping = l['hopping']
                lead_phi = l['phi']
                lead_r = l['reverse']
                sym = kwant.TranslationalSymmetry(self.lattice.vec(lead_vector))
                a = orthogVecSlope(self.lattice.vec(lead_vector))
                lead_up = kwant.Builder(sym)
                lead_down = kwant.Builder(sym)
                print(lead_range)
                lead_up[self.lattice.shape(lambda pos: lead_range[0]  < pos[1] + pos[0] * a < lead_range[1], lead_offset)] = partial(onSiteFunction, self, lead_pot, 1/2, lead_phi)
                lead_down[self.lattice.shape(lambda pos: lead_range[0]  < pos[1] + pos[0] * a < lead_range[1], lead_offset)] = partial(onSiteFunction, self, lead_pot, -1/2, lead_phi)
                lead_up[self.neighbors] = partial(hoppingFunction, self, lead_hopping, lead_phi)
                lead_down[self.neighbors] = partial(hoppingFunction, self, lead_hopping, lead_phi)            
                lead_up.eradicate_dangling()
                lead_down.eradicate_dangling()

                self.system_up.attach_lead(lead_up)
                self.system_down.attach_lead(lead_down)

                if lead_r:
                    self.system_up.attach_lead(lead_up.reversed())
                    self.system_down.attach_lead(lead_down.reversed())

        else:
            for l in self.leads:
                lead_range = l['range']
                lead_vector = l['symmetry']
                lead_offset = l['offset']
                lead_pot = l['potential']
                lead_hopping = l['hopping']
                lead_phi = l['phi']
                lead_r = l['reverse']
                sym = kwant.TranslationalSymmetry(self.lattice.vec(lead_vector))
                a = orthogVecSlope(self.lattice.vec(lead_vector))
                lead = kwant.Builder(sym)
                lead[self.lattice.shape(lambda pos: lead_range[0]  < pos[1] + pos[0] * a < lead_range[1], lead_offset)] = lead_pot
                lead[self.neighbors] = partial(hoppingFunction, self, lead_hopping, lead_phi)
                lead.eradicate_dangling()
                self.system.attach_lead(lead)

                if lead_r:
                    self.system.attach_lead(lead.reversed())


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
        if self.spin_dep:
            self.pre_system_up = self.system_up
            self.pre_system_down = self.system_down
        
            self.system_up = self.system_up.finalized()
            self.system_down = self.system_down.finalized()
        else:
            self.pre_system = self.system
            self.system = self.system.finalized()

    def getPreSystem(self):
        if self.spin_dep:
            return self.pre_system_up, self.pre_system_down
        else:
            return self.pre_system

    def getSystem(self):
        if self.spin_dep:
            return self.system_up, self.system_down
        else:
            return self.system

    def visualizeSystem(self, args={}):
        if self.spin_dep:
            return kwant.plot(self.system_up, site_lw=0.1, colorbar=False, **args, show=False)
        else:
            return kwant.plot(self.system, site_lw=0.1, colorbar=False, **args, show=False)

    def diagonalize(self, args={}):
        # Compute some eigenvalues of the closed system
        if self.spin_dep:
            sparse_mat_up = self.system_up.hamiltonian_submatrix(sparse=True, **args)
            sparse_mat_down = self.system_down.hamiltonian_submatrix(sparse=True, **args)
            return sla.eigs(sparse_mat_up), sla.eigs(sparse_mat_down) 
        else:
            sparse_mat = self.system.hamiltonian_submatrix(sparse=True, **args)
            return sla.eigs(sparse_mat)

    def getConductance(self, energies, start_lead_id, end_lead_id):
        # Compute transmission as a function of energy
        if self.spin_dep:
            data_up, data_down = [], []
            for energy in energies:
                smatrix_up = kwant.smatrix(self.system_up, energy)
                smatrix_down = kwant.smatrix(self.system_down, energy)
                data_up.append(smatrix_up.transmission(start_lead_id, end_lead_id))
                data_down.append(smatrix_down.transmission(start_lead_id, end_lead_id))
            return data_up, data_down
        else:
            data = []
            for energy in energies:
                smatrix = kwant.smatrix(self.system, energy)
                data.append(smatrix.transmission(start_lead_id, end_lead_id))
            return data

    def getCurrent(self, energies):
        if self.spin_dep:
            pass

    def getBandStructure(self, lead_id, momenta=np.linspace(-np.pi, np.pi, 256)):
        """
        Get the band structure of a certain lead.

        Parameters
        ----------
        lead_id : The id of the lead the band structure will be calculated.
        momenta : The values of momenta for which you would like the band structure. Default: np.linspace(-np.pi, np.pi, 256)

        Returns
        -------
        A tuple with length 2 or three. Zeroth element is momenta and the rest are the energies for that momenta. For spin polarized systems it returns the bands for spin-up and spin-down (length 3 tuple).
        """
        if self.spin_dep:
            bands_up = kwant.physics.Bands(self.system_up.leads[lead_id].finalized())
            bands_down = kwant.physics.Bands(self.system_down.leads[lead_id].finalized())
            energies_up = [bands_up(k) for k in momenta]
            energies_down = [bands_down(k) for k in momenta]
            return momenta, energies_up, energies_down         
        else:
            bands = kwant.physics.Bands(self.system.leads[lead_id].finalized())
            energies = [bands(k) for k in momenta]
            return momenta, energies

    def getEnergyRange(self):
        """
        Find the energy range that must be integrated over to calcuate the current.

        Returns
        -------
        A list of length 2 with the zeroth element being the minimum and the first element being the maximum.
        """
        if self.spin_dep:
            e_ups = [], e_downs = []
            for i, l in enumerate(self.system_up.leads):
                m, e_up, e_down = self.getBandStructure(i)
                e_ups.append(e_up)
                e_downs.append(e_down)
            return [np.min(e_ups + e_downs), np.max(e_ups + e_downs)]
        else:
            es = []
            for i, l in enumerate(self.system.leads):
                m, e = self.getBandStructure(i)
                es.append(e)
            return[np.min(np.array(es)), np.max(np.flatten(es))]

    def getWaveFunction(self, lead_id, energy=-1):
        if self.spin_dep:
            return kwant.wave_function(self.system_up, energy)(lead_id), kwant.wave_function(self.system_down, energy)(lead_id)
        else:
            return kwant.wave_function(self.system, energy)(lead_id)

    # currently not working
    def plotWaveFunction(self, lead_id, energy=0., cmap=cmocean.cm.dense):
        return kwant.plotter.map(self.system, np.absolute(self.getWaveFunction(lead_id, energy)[0])**2, oversampling=10, cmap=cmap)

    def plotCurrentDensity(self, lead_id, energy=0., args={}):
        if self.spin_dep:
            pass
        else:
            J = kwant.operator.Current(self.system)
            current = np.sum(J(p) for p in self.getWaveFunction(lead_id, energy))
            return kwant.plotter.current(self.system, current, cmap=cmocean.cm.dense, **args)

    def getNSites(self):
        if self.spin_dep:
            return len(list(self.pre_system_up.sites()))
        else:
            return len(list(self.pre_system.sites()))

    def getDOS(self):
        return kwant.kpm.SpectralDensity(self.system)()


