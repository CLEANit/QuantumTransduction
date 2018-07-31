#!/usr/bin/env python

import kwant
import scipy.sparse.linalg as sla
import numpy as np
import time
from functools import partial
import cmocean

from .helper import *
import coloredlogs, verboselogs

# create logger
coloredlogs.install(level='INFO')
logger = verboselogs.VerboseLogger('QMT::Structure ')

def hoppingFunction(self, t, phi, direction, site1, site2):
    """
    This is a function which returns the hopping value when a magnetic field is being applied.

    For the magnetic field: A value of 1 phi = 789436.238138607 Tesla!

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


    return t * np.exp(-1j * np.pi * (p2[0] - p1[0]) * (p2[1] + p1[1]) * B)

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
    B_in_T = 6.62607004e-34 * B * 10e20 / 1.60217662e-19
    B_in_G = B_in_T * 1e4
    return pot + 2.0 * 0.579e-8 * spin * B_in_G

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
        self.grid_size = 128
        self.device = parser.getDevice()
        self.leads = parser.getLeads()
        self.body = self.device['body']
        self.spin_dep = parser.spinDependent()

        self.build()

        self.mask = self.parser.getMaskFunction()
        if self.mask is not None:
            if self.spin_dep:
                self.system_up = self.applyMask(self.mask, self.system_up)
                self.system_down = self.applyMask(self.mask, self.system_down)
            else:
                self.system = self.applyMask(self.mask, self.system)
        else:
            # there is no mask to apply
            pass
            
        self.attachLeads()


        self.finalize()

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


        for shape, offset, hopping, potential, direction in zip(self.device['shapes'], self.device['offsets'], self.device['hoppings'], self.device['potentials'], self.device['directions']):
            
            # if we want to consider spin dependent transport
            if self.spin_dep:
                self.system_up[self.lattice.shape(shape, offset)] = partial(onSiteFunction, self, potential, 1/2, self.parser.getPhi())
                self.system_down[self.lattice.shape(shape, offset)] = partial(onSiteFunction, self, potential, -1/2, self.parser.getPhi())
                
                self.neighbors = self.lattice.neighbors()
                
                self.system_up[self.neighbors] = partial(hoppingFunction, self, hopping, self.parser.getPhi(), direction)
                self.system_down[self.neighbors] = partial(hoppingFunction, self, hopping, self.parser.getPhi(), direction)

            else:
                self.system[self.lattice.shape(shape, offset)] = potential
                self.neighbors = self.lattice.neighbors()
                self.system[self.neighbors] = partial(hoppingFunction, self, hopping, self.parser.getPhi(), direction)

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
                lead_r = l['reverse']
                lead_shift = l['shift']
                lead_direction = l['direction']
                sym = kwant.TranslationalSymmetry(self.lattice.vec(lead_vector))
                a = orthogVecSlope(self.lattice.vec(lead_vector))
                lead_up = kwant.Builder(sym)
                lead_down = kwant.Builder(sym)
                lead_up[self.lattice.shape(lambda pos: lead_range[0]  <= pos[1] - lead_shift[1] + (pos[0] - lead_shift[0]) * a <= lead_range[1], lead_offset)] = partial(onSiteFunction, self, lead_pot, 1/2, self.parser.getPhi())
                lead_down[self.lattice.shape(lambda pos: lead_range[0]  <= pos[1] - lead_shift[1] + (pos[0] - lead_shift[0]) * a <= lead_range[1], lead_offset)] = partial(onSiteFunction, self, lead_pot, -1/2, self.parser.getPhi())
                lead_up[self.neighbors] = partial(hoppingFunction, self, lead_hopping, self.parser.getPhi(), lead_direction)
                lead_down[self.neighbors] = partial(hoppingFunction, self, lead_hopping, self.parser.getPhi(), lead_direction)            
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
                lead_r = l['reverse']
                lead_shift = l['shift']
                lead_direction = l['direction']
                sym = kwant.TranslationalSymmetry(self.lattice.vec(lead_vector))
                a = orthogVecSlope(self.lattice.vec(lead_vector))
                lead = kwant.Builder(sym)
                lead[self.lattice.shape(lambda pos: lead_range[0]  <= pos[1] - lead_shift[1] + (pos[0] - lead_shift[0]) * a <= lead_range[1], lead_offset)] = lead_pot
                lead[self.neighbors] = partial(hoppingFunction, self, lead_hopping, self.parser.getPhi(), lead_direction)
                lead.eradicate_dangling()
                self.system.attach_lead(lead)

                if lead_r:
                    self.system.attach_lead(lead.reversed())


    def applyMask(self, mask, system):
        tags = []
        positions = []
        sites = []
        for s, v in system.site_value_pairs():
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
            del system[sites[int(elem)]]
        system.eradicate_dangling()

        return system

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

    def getConductance(self, lead_in, lead_out, energies=None):
        """
        Compute the conductance between 2 leads.

        Parameters
        ----------
        lead_in : The id of the lead you would like to inject electrons.
        lead_out : The id of the lead you measuring the transmission through.
        energies : The range of energies that must be integrated over. By default, the energies are found from the minimum and maximum values in the band structure.

        Returns
        -------
        The range of energies and either one or two (spin-dependent) vectors of conductance. i.e.

        (energies, conductances) 

        or

        (energies, conductances_up, conductances_down)

        """
        if energies == None:
            energy_range = self.getEnergyRange()
            energies = np.linspace(energy_range[0], energy_range[1], self.grid_size)

        if self.spin_dep:
            data_up, data_down = [], []
            for energy in energies:
                smatrix_up = kwant.smatrix(self.system_up, energy)
                smatrix_down = kwant.smatrix(self.system_down, energy)
                data_up.append(smatrix_up.transmission(lead_in, lead_out))
                data_down.append(smatrix_down.transmission(lead_in, lead_out))
            return energies, data_up, data_down
        else:
            data = []
            for energy in energies:
                smatrix = kwant.smatrix(self.system, energy)
                data.append(smatrix.transmission(lead_in, lead_out))
            return energies, data

    def getCurrent(self, lead_in, lead_out, avg_chem_pot=1.0):
        """
        Compute the current between 2 leads.

        Parameters
        ----------
        lead_in : The id of the lead you are injecting electrons.
        lead_out : The id of the lead you would like to find the transmission through.
        avg_chem_pot : The average chemical potential difference. Default: 1.0. It is common practice to set this to the hopping energy.

        Returns
        -------
        Either one or two (spin-dependent) values of current i.e.

        current

        or 

        current_up, current_down

        """

        bias = self.parser.getBias()
        kb_T = self.parser.getKBT()

        if self.spin_dep:
            e, cond_up, cond_down = self.getConductance(lead_in, lead_out)
            de = e[1] - e[0]
            mu_left = bias / 2.0 + avg_chem_pot
            mu_right = -bias / 2.0 + avg_chem_pot
            diff_fermi = vectorizedFermi(e, mu_left, kb_T) - vectorizedFermi(e, mu_right, kb_T)
            return de * np.sum(cond_up * diff_fermi), de * np.sum(cond_down * diff_fermi)

        else:
            e, cond = self.getConductance(lead_in, lead_out)
            de = e[1] - e[0]
            mu_left = bias / 2.0 + avg_chem_pot
            mu_right = -bias / 2.0 + avg_chem_pot
            diff_fermi = vectorizedFermi(e, mu_left, kb_T) - vectorizedFermi(e, mu_right, kb_T)
            return de * np.sum(cond * diff_fermi)



    def getBandStructure(self, lead_id, momenta=None):
        """
        Get the band structure of a certain lead.

        Parameters
        ----------
        lead_id : The id of the lead the band structure will be calculated.
        momenta : The values of momenta for which you would like the band structure. Default: np.linspace(-np.pi, np.pi, self.grid_size)

        Returns
        -------
        A tuple with length 2 or three. Zeroth element is momenta and the rest are the energies for that momenta. For spin polarized systems it returns the bands for spin-up and spin-down (length 3 tuple).
        """
        if momenta == None:
            momenta = np.linspace(-np.pi, np.pi, self.grid_size)

        if self.spin_dep:
            bands_up = kwant.physics.Bands(self.system_up.leads[lead_id])
            bands_down = kwant.physics.Bands(self.system_down.leads[lead_id])
            energies_up = [bands_up(k) for k in momenta]
            energies_down = [bands_down(k) for k in momenta]
            return momenta, energies_up, energies_down         
        else:
            bands = kwant.physics.Bands(self.system.leads[lead_id])
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
            mins = []
            maxs = []
            for i, l in enumerate(self.system_up.leads):
                m, e_up, e_down = self.getBandStructure(i)
                mins.append(np.min(np.array(e_up).flatten()))
                maxs.append(np.max(np.array(e_up).flatten()))
                mins.append(np.min(np.array(e_down).flatten()))
                maxs.append(np.max(np.array(e_down).flatten()))
            return [min(mins), max(maxs)]
        else:
            mins = []
            maxs = []
            for i, l in enumerate(self.system.leads):
                m, e = self.getBandStructure(i)
                mins.append(np.min(np.array(e).flatten()))
                maxs.append(np.max(np.array(e).flatten()))
            return [min(mins), max(maxs)]

    def getDOS(self, energies=None):
        """
        Get the density of states for the system.

        Parameters
        ----------
        energies : An array of energies that we would like to calculate the LDOS at. Default is the minimum and maximum found from the bandstructure.

        Returns
        -------
        Energies and DOS in the form of a tuple. If spin-dependent calculations are specified, it returns energies, DOS for spin up, DOS for spin down.
        """
        if energies == None:
            energy_range = self.getEnergyRange()
            energies = np.linspace(energy_range[0], energy_range[1], self.grid_size)

        if self.spin_dep:
            es, DOS_up, DOS_down = [], [], []
            for e in energies:
                # sometimes the ldos function returns an error for a certain value of energy
                # -- we therefore must use a try-except statement
                try:
                    LDOS_up = kwant.ldos(self.system_up, e)
                    LDOS_down = kwant.ldos(self.system_down, e)
                    es.append(e)
                    # integrate the ldos over all space
                    DOS_up.append(np.sum(LDOS_up))
                    DOS_down.append(np.sum(LDOS_down))
                except:
                    pass
            return es, DOS_up, DOS_down
        else:
            es, DOS = [], []
            for e in energies:
                # sometimes the ldos function returns an error for a certain value of energy
                # -- we therefore must use a try-except statement
                # try:
                LDOS = kwant.ldos(self.system, e)
                es.append(e)
                # integrate the ldos over all space
                DOS.append(np.sum(LDOS))
                # except:
                #     pass
            return es, DOS

    def getValleyPolarizedConductance(self, energy, lead_start=0, lead_end=1, K_prime_range=(-np.inf, -1e8), K_range=(0, np.inf), velocities='left_moving'):
        """
        Get the valley-polarized conductances for a given energy between two leads. Note: This function only makes sense when
        the bandstructure has two valleys in it. An example is zig-zag edged graphene nanoribbons.

        Parameters
        ----------
        energy : A value of energy.
        lead_start : An integer of the lead where electrons are injected.
        lead_end : An integer of the lead where electrons are transmitting through.
        K_prime_range : A tuple of length 2 which defines the range where K' would be the polarization. Default: (-np.inf, -1e8)
        K_range : A tuple of length 2 which defines the range where K would be the polarization. Default (0, np.inf)
        velocities : If 'left_moving', we only consider velocities >= 0. If 'right_moving', velocities < 0. Default: 'left_moving'

        Returns
        -------
        A tuple of length 2. First element is for K', second for K.
        """

        if self.spin_dep:
            logger.error('You are trying to compute the Valley Conductances for Spin-Polarized calculations. This is currently not supported.')
            exit(-1)

        smatrix = kwant.smatrix(syst, energy)
        if velocities == 'left_moving':
            positives = np.where(smatrix.lead_info[lead_start].velocities >= 0)[0]
        elif velocities == 'right_moving':
            positives = np.where(smatrix.lead_info[lead_start].velocities < 0)[0]
        else:
            logger.error("You have defined the direction of the velocities wrong. It is either 'left_moving' or 'right_moving'.")

        momentas = smatrix.lead_info[lead_start].momenta[positives]
        K_prime_indices = np.where(momentas >= K_prime_range[0] and momentas <= K_prime_range[1])[0]
        K_indices = np.where(momentas >= K_range[0] and momentas <= K_range[1])[0]
        submatrix = smatrix.submatrix(lead_end, lead_start)
        K_prime_T = np.sum(np.absolute(submatrix[:, K_prime_indices])**2) 
        K_T = np.sum(np.absolute(submatrix[:, K_indices])**2)
        return (K_prime_T, K_T)



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

    def getChromosome(self):
        chromosome = []
        for gene in self.parser.getGenes():
            val = getFromDict(self.parser.getConfig(), gene['path'])
            chromosome.append(val)
        return chromosome

