#!/usr/bin/env python

import kwant
import scipy.sparse.linalg as sla

class Model:
    def __init__(   self, 
                    shape,
                    potential,
                    lead_shapes,
                    lead_vectors,
                    lead_offsets,
                    lead_potentials,
                    shape_offset=(0., 0.),
                    lattice_type='graphene',
                    lattice_const=1.0,
                    t=1.0,
                 ):


        # type of lattice construction
        if lattice_type == 'graphene':
            self.lattice = kwant.lattice.honeycomb(lattice_const)
            self.a, self.b = self.lattice.sublattices

        # define the class parameters
        self.shape = shape
        self.potential = potential

        assert len(lead_shapes) == len(lead_vectors) == len(lead_offsets)
        
        self.lead_shapes = lead_shapes
        self.lead_vectors = lead_vectors
        self.lead_offsets = lead_offsets
        self.lead_potentials = lead_potentials

        self.shape_offset = shape_offset
        self.lattice_type = lattice_type
        self.lc = lattice_const 
        self.t = t

        # initialize the builder
        self.system = kwant.Builder()

        # build the structure
        self.build()


    def build(self):
        self.hoppings = self.lattice.neighbors()
        self.system[self.lattice.shape(self.shape, self.shape_offset)] = self.potential
        self.system[self.hoppings] = -self.t

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

    def getSystem(self):
        return self.system

    def getLeads(self):
        return self.leads

    def family_colors(self, site):
        return 0 if site.family == self.a else 1

    def visualizeSystem(self):
        return kwant.plot(self.system, site_color=self.family_colors, site_lw=0.1, colorbar=False)

    def finalize(self):
        self.system = self.system.finalized()

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