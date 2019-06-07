#!/usr/bin/env python

import kwant
import scipy.sparse.linalg as sla
import numpy as np
import time
from functools import partial
import cmocean
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import tinyarray as ta
from sklearn.neural_network import MLPRegressor
from scipy import signal
from scipy.ndimage import measurements, gaussian_filter
import os
from .helper import *
import coloredlogs, verboselogs
from skimage.morphology import binary_erosion, binary_dilation

# create logger
coloredlogs.install(level='INFO')
logger = verboselogs.VerboseLogger('qmt::structure ')

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

def onSiteFunction(self, pot, spin, phi, lead, site):
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
    pnj_config = self.parser.getPNJunction()
    
    if type(pot) == float:
        pot = np.array([[pot]])
    else:
        pot = np.array(pot)

    if pnj_config['turn_on'] == True and lead == False:
        # pn-junction stuff
        ###################################################
        if pointInHull(site.pos, self.hull):
            np.fill_diagonal(pot , pot.diagonal() + pnj_config['p-potential'])
        else:
            np.fill_diagonal(pot , pot.diagonal() + pnj_config['n-potential'])
        ###################################################



        # magnetic field stuff
        ###################################################
        lattice_vectors = self.parser.getLatticeVectors()

        B = phi / (lattice_vectors[0][0] * lattice_vectors[1][1] - lattice_vectors[0][1] * lattice_vectors[1][0])
        B_in_T = 6.62607004e-34 * B * 10e20 / 1.60217662e-19
        B_in_G = B_in_T * 1e4
        np.fill_diagonal(pot, pot.diagonal() + 2.0 * 0.579e-8 * spin * B_in_G)
        ###################################################

        return ta.array(pot)
    
    else:
        lattice_vectors = self.parser.getLatticeVectors()

        B = phi / (lattice_vectors[0][0] * lattice_vectors[1][1] - lattice_vectors[0][1] * lattice_vectors[1][0])
        B_in_T = 6.62607004e-34 * B * 10e20 / 1.60217662e-19
        B_in_G = B_in_T * 1e4
        np.fill_diagonal(pot, pot.diagonal() + 2.0 * 0.579e-8 * spin * B_in_G)

        return pot 

class Structure:
    """
    This is the main class which stores all of the information needed for kwant as well as functions to make it easier to use kwant (not that it is too difficult).

    Parameters
    ----------
    parser : This takes the YAML configuration file which is read in as a dictionary.

    """
    def __init__(   self,
                    parser,
                    identifier,
                    parents
                 ):
        # define the class parameters
        self.parser = parser
        self.grid_size = 128
        self.device = parser.getDevice()
        self.leads = parser.getLeads()
        self.body = self.device['body']
        self.spin_dep = parser.spinDependent()
        self.identifier = identifier
        self.parents = parents

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

        self.pnj_config = self.parser.getPNJunction()

        if self.pnj_config['turn_on'] == True:
            # print(self.pnj_config['points'])
            self.hull = ConvexHull(self.pnj_config['points'])

        if self.parser.getGenerator()['turn_on']:
            self.system = self.policyMask(self.system)

        self.attachLeads()

        self.finalize()

        self.finished_calculations = dict()

    def build(self):

        if self.parser.config['System']['pre_defined'] == 'MoS2':

            # lattice constant
            a = 3.19
            # hopping energy
            eps_1 = 1.046
            eps_2 = 2.104
            t0 = -0.184
            t1 = 0.401
            t2 = 0.507
            t11 = 0.218
            t12 = 0.338
            t22 = 0.057
            rt3 = np.sqrt(3)

            h0 = np.array([[eps_1,0,0],[0,eps_2,0],[0,0,eps_2]])

            h1 = ta.array([[ t0, -t1,   t2],
                  [ t1, t11, -t12],
                  [ t2, t12,  t22]])

            h2 = ta.array([[                    t0,     1/2 * t1 + rt3/2 * t2,     rt3/2 * t1 - 1/2 * t2],
                  [-1/2 * t1 + rt3/2 * t2,     1/4 * t11 + 3/4 * t22, rt3/4 * (t11 - t22) - t12],
                  [-rt3/2 * t1 - 1/2 * t2, rt3/4 * (t11 - t22) + t12,     3/4 * t11 + 1/4 * t22]])

            h3 = ta.array([[                    t0,    -1/2 * t1 - rt3/2 * t2,     rt3/2 * t1 - 1/2 * t2],
                  [ 1/2 * t1 - rt3/2 * t2,     1/4 * t11 + 3/4 * t22, rt3/4 * (t22 - t11) + t12],
                  [-rt3/2 * t1 - 1/2 * t2, rt3/4 * (t22 - t11) - t12,     3/4 * t11 + 1/4 * t22]])
            
            v2 = np.array((a / 2., np.sqrt(3.) * a / 2.0))
            v1 = np.array((a,0))
            self.lattice = kwant.lattice.general([v1, v2], [(0, 0)], norbs=1 )
            lat = self.lattice.sublattices[0]

            self.system = kwant.Builder()

            for shape, offset, hopping, potential, direction in zip(self.device['shapes'], self.device['offsets'], self.device['hoppings'], self.device['potentials'], self.device['directions']):
                
                # if we want to consider spin dependent transport
                self.system[self.lattice.shape(shape, offset)] = partial(onSiteFunction, self, h0, 0., self.parser.getPhi(), False)
                self.system[kwant.builder.HoppingKind((1,0), lat, lat)] =  h1
                self.system[kwant.builder.HoppingKind((0,-1), lat, lat)] = simTransformX(4 * np.pi / 3, h1)
                self.system[kwant.builder.HoppingKind((1,-1), lat, lat)] = reflectZ(simTransformX(np.pi / 3, h1))
                

            self.system.eradicate_dangling()


        elif self.parser.config['System']['pre_defined'] == 'graphene':
            pass

        elif self.parser.config['System']['pre_defined'] == None:

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
                    self.system_up[self.lattice.shape(shape, offset)] = partial(onSiteFunction, self, potential, 1/2, self.parser.getPhi(), False)
                    self.system_down[self.lattice.shape(shape, offset)] = partial(onSiteFunction, self, potential, -1/2, self.parser.getPhi(), False)
                    
                    self.neighbors = self.lattice.neighbors()
                    
                    self.system_up[self.neighbors] = partial(hoppingFunction, self, hopping, self.parser.getPhi(), direction)
                    self.system_down[self.neighbors] = partial(hoppingFunction, self, hopping, self.parser.getPhi(), direction)

                else:
                    self.system[self.lattice.shape(shape, offset)] = partial(onSiteFunction, self, potential, 0., self.parser.getPhi(), False)
                    self.neighbors = self.lattice.neighbors()
                    self.system[self.neighbors] = partial(hoppingFunction, self, hopping, self.parser.getPhi(), direction)

            if self.spin_dep:
                self.system_up.eradicate_dangling()
                self.system_down.eradicate_dangling()
            else:
                self.system.eradicate_dangling()

    def attachLeads(self):

        if self.parser.config['System']['pre_defined'] == 'MoS2':

            # lattice constant
            a = 3.19
            # hopping energy
            eps_1 = 1.046
            eps_2 = 2.104
            t0 = -0.184
            t1 = 0.401
            t2 = 0.507
            t11 = 0.218
            t12 = 0.338
            t22 = 0.057
            rt3 = np.sqrt(3)

            h0 = np.array([[eps_1,0,0],[0,eps_2,0],[0,0,eps_2]])

            h1 = ta.array([[ t0, -t1,   t2],
                  [ t1, t11, -t12],
                  [ t2, t12,  t22]])

            h2 = ta.array([[                    t0,     1/2 * t1 + rt3/2 * t2,     rt3/2 * t1 - 1/2 * t2],
                  [-1/2 * t1 + rt3/2 * t2,     1/4 * t11 + 3/4 * t22, rt3/4 * (t11 - t22) - t12],
                  [-rt3/2 * t1 - 1/2 * t2, rt3/4 * (t11 - t22) + t12,     3/4 * t11 + 1/4 * t22]])

            h3 = ta.array([[                    t0,    -1/2 * t1 - rt3/2 * t2,     rt3/2 * t1 - 1/2 * t2],
                  [ 1/2 * t1 - rt3/2 * t2,     1/4 * t11 + 3/4 * t22, rt3/4 * (t22 - t11) + t12],
                  [-rt3/2 * t1 - 1/2 * t2, rt3/4 * (t22 - t11) - t12,     3/4 * t11 + 1/4 * t22]])
            lat = self.lattice.sublattices[0]

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
                lead[self.lattice.shape(lambda pos: lead_range[0]  <= pos[1] - lead_shift[1] + (pos[0] - lead_shift[0]) * a <= lead_range[1], lead_offset)] = partial(onSiteFunction, self, h0, 0., self.parser.getPhi(), False)
                lead[kwant.builder.HoppingKind((1,0), lat, lat)] =  h1
                lead[kwant.builder.HoppingKind((0,-1), lat, lat)] = simTransformX(4 * np.pi / 3, h1)
                lead[kwant.builder.HoppingKind((1,-1), lat, lat)] = reflectZ(simTransformX(np.pi / 3, h1))

                lead.eradicate_dangling()
                self.system.attach_lead(lead)

                if lead_r:
                    self.system.attach_lead(lead.reversed())        
        else:
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
                    lead_up[self.lattice.shape(lambda pos: lead_range[0]  <= pos[1] - lead_shift[1] + (pos[0] - lead_shift[0]) * a <= lead_range[1], lead_offset)] = partial(onSiteFunction, self, lead_pot, 1/2, self.parser.getPhi(), True)
                    lead_down[self.lattice.shape(lambda pos: lead_range[0]  <= pos[1] - lead_shift[1] + (pos[0] - lead_shift[0]) * a <= lead_range[1], lead_offset)] = partial(onSiteFunction, self, lead_pot, -1/2, self.parser.getPhi(), True)
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
                    lead[self.lattice.shape(lambda pos: lead_range[0]  <= pos[1] - lead_shift[1] + (pos[0] - lead_shift[0]) * a <= lead_range[1], lead_offset)] = partial(onSiteFunction, self, lead_pot, 0., self.parser.getPhi(), True)
                    lead[self.neighbors] = partial(hoppingFunction, self, lead_hopping, self.parser.getPhi(), lead_direction)
                    lead.eradicate_dangling()
                    self.system.attach_lead(lead)

                    if lead_r:
                        self.system.attach_lead(lead.reversed())

    def policyMask(self, system):
        nns = []
        neighborhoods = []
        self.system_colours = {}
        pnj_config = self.parser.getPNJunction()
        tags = []
        poss = []
        for s, v in system.site_value_pairs():
            if self.body(s.pos):
                tags.append(s.tag)
                poss.append(s.pos)
                neighborhood = {}
                for n in self.system.neighbors(s):
                    val = self.system[n]
                    neighborhood[n.index] = np.mean(val(n))
                    for nn in self.system.neighbors(n):
                        val = self.system[nn]
                        neighborhood[nn.index] = np.mean(val(nn))

                neighborhoods.append((s, np.array(list(neighborhood.values()))))
                nns.append(len(neighborhood))

        max_vec_size = np.max(nns)

        # get the ANN
        generator_params = self.parser.getGenerator()

        # create the ANN if we need it
        if self.parser.policy_mask is None:
            self.parser.policy_mask = MLPRegressor(hidden_layer_sizes=[max_vec_size] + generator_params['neurons'] + [2])
            self.parser.policy_mask._random_state = np.random.RandomState(np.random.randint(2**32))
            self.parser.policy_mask._initialize(np.empty((1, 2)), [max_vec_size, 128, 2])

        for s, neighborhood in neighborhoods:
            if self.body(s.pos):
                input_vec = np.zeros((1, max_vec_size))
                input_vec[0, :neighborhood.shape[0]] += neighborhood[:]
                output_vec = self.parser.policy_mask.predict(input_vec)[0, :]
                output_vec = np.exp(output_vec) / np.sum(np.exp(output_vec))
                choice = np.random.choice([0, 1], p=output_vec)

                self.system_colours[s] = choice

                for nn in self.system.neighbors(s):
                    self.system_colours[nn] = choice

                    for nnn in self.system.neighbors(nn):
                        self.system_colours[nnn] = choice

        # return system
        # import scipy
        bin_rep = self.getBinaryRepresentation(system, policyMask=True)
        bin_rep = gaussian_filter(bin_rep, 2)
        bin_rep = np.round(bin_rep)
        bin_rep = binary_erosion(bin_rep, selem=np.ones((3,3)))
        bin_rep = binary_dilation(bin_rep, selem=np.ones((3,3)))

        tags = np.array(tags)
        poss = np.array(poss)

        min_pos_sx = np.min(poss[:,0])
        min_pos_sy = np.min(poss[:,1])
        max_pos_sx = np.max(poss[:,0])
        max_pos_sy = np.max(poss[:,1])
        
        min_tag_sx = np.min(tags[:,0])
        min_tag_sy = np.min(tags[:,1])
        max_tag_sx = np.max(tags[:,0])
        max_tag_sy = np.max(tags[:,1])

        image_size = (max_tag_sx - min_tag_sx + 1, max_tag_sy - min_tag_sy + 1)
        dx = (max_pos_sx - min_pos_sx) / (image_size[0] - 1)
        dy = (max_pos_sy - min_pos_sy) / (image_size[1] - 1)

        for s, v in system.site_value_pairs():
            if self.body(s.pos):
                index_x = int((s.pos[0] - min_pos_sx) / dx)
                index_y = int((s.pos[1] - min_pos_sy) / dy)
                choice = bin_rep[index_x, index_y]
                try:
                    pot = np.array(system[s](s))
                except:
                    pot = np.array(system[s])

                if choice:
                    np.fill_diagonal(pot , pot.diagonal() + pnj_config['p-potential'])
                else:
                    np.fill_diagonal(pot , pot.diagonal() + pnj_config['n-potential'])
                self.system_colours[s] = choice
                system[s] = ta.array(pot)

        # plt.imshow(np.rot90(bin_rep))
        # # plt.colorbar()
        # plt.show()
        return system
        # bin_rep_fft = np.fft.fft2(bin_rep)
        # p_spec = np.absolute(bin_rep_fft)**2
        # labelled_arr, num_clusters = measurements.label(bin_rep)
        # kx = np.fft.fftfreq(bin_rep.shape[0], d=1 / bin_rep.shape[0])
        # ky = np.fft.fftfreq(bin_rep.shape[1], d=1 / bin_rep.shape[1])
        # k_x, k_y = np.meshgrid(kx, ky)
        # plt.imshow(bin_rep)
        # # plt.colorbar()
        # plt.show()
        # plt.pcolormesh(k_x, k_y, np.log(p_spec.T))
        # # plt.colorbar()
        # plt.show()

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

    def getBinaryRepresentation(self, system, policyMask=False):
        """
        Return a binary image that describes the pn-junction in image form.
        """
        if not policyMask:
            tags = []
            positions = []
            for s, v in system.site_value_pairs():
                tags.append(s.tag)
                positions.append(s.pos)
            tags = np.array(tags)
            positions = np.array(positions)
            
            min_tag_sx = np.min(tags[:,0])
            min_tag_sy = np.min(tags[:,1])
            max_tag_sx = np.max(tags[:,0])
            max_tag_sy = np.max(tags[:,1])

            image = np.zeros((max_tag_sx - min_tag_sx + 1, max_tag_sy - min_tag_sy + 1))

            for t, p in zip(tags, positions):
                if pointInHull(p, self.hull):
                    image[t[0] - min_tag_sx][t[1] - min_tag_sy] = self.parser.config['System']['Junction']['body']['pn-junction']['n-potential']
                else:
                    image[t[0] - min_tag_sx][t[1] - min_tag_sy] = self.parser.config['System']['Junction']['body']['pn-junction']['p-potential']
            return image
        else:
            tags = []
            poss = []
            for s, v in system.site_value_pairs():
                if self.body(s.pos):
                    tags.append(s.tag)
                    poss.append(s.pos)
            
            tags = np.array(tags)
            poss = np.array(poss)

            min_pos_sx = np.min(poss[:,0])
            min_pos_sy = np.min(poss[:,1])
            max_pos_sx = np.max(poss[:,0])
            max_pos_sy = np.max(poss[:,1])
            
            min_tag_sx = np.min(tags[:,0])
            min_tag_sy = np.min(tags[:,1])
            max_tag_sx = np.max(tags[:,0])
            max_tag_sy = np.max(tags[:,1])

            image_size = (max_tag_sx - min_tag_sx + 1, max_tag_sy - min_tag_sy + 1)
            dx = (max_pos_sx - min_pos_sx) / image_size[0]
            dy = (max_pos_sy - min_pos_sy) / image_size[1]

            image = np.zeros(image_size)
            for s, v in system.site_value_pairs():
                try:
                    index_x = int((s.pos[0] - min_pos_sx) / dx)
                    index_y = int((s.pos[1] - min_pos_sy) / dy)
                    image[index_x, index_y] = self.system_colours[s]
                except KeyError:
                    pass
                except IndexError:
                    pass
            return image

    def visualizeSystem(self, args={}):
        """
        Create a plot to visualize the constructed system.

        Returns
        -------
        A pyplot object.
        """
        if self.spin_dep:
            return kwant.plot(self.pre_system_up, site_lw=0.1, colorbar=False, show=False, **args)
        else:

            if self.pnj_config['turn_on'] == True:
                def siteColours(site):
                    # print(list(self.pre_system.sites())[site])
                    if pointInHull(site.pos, self.hull):
                        return cmocean.cm.deep(0.9)
                    else:
                        return cmocean.cm.deep(0.1)
                return kwant.plot(self.pre_system, site_lw=0.1, lead_site_lw=0, colorbar=False, site_color=siteColours, show=True, **args)
            elif self.parser.getGenerator()['turn_on']:
                def siteColours(site):
                    # print(list(self.pre_system.sites())[site])
                    try:
                        if self.system_colours[site]:
                            return cmocean.cm.deep(0.9)
                        else:
                            return cmocean.cm.deep(0.1)
                    except:
                        return cmocean.cm.deep(0.5)
                return kwant.plot(self.pre_system, site_lw=0.1, lead_site_lw=0, colorbar=False, site_color=siteColours, show=True, **args)            
            else:
                return kwant.plot(self.pre_system, site_lw=0.1, colorbar=False, show=False, **args)

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
                data_up.append(smatrix_up.transmission(lead_out, lead_in))
                data_down.append(smatrix_down.transmission(lead_out, lead_in))
            return energies, data_up, data_down
        else:
            data = []
            for energy in energies:
                smatrix = kwant.smatrix(self.system, energy)
                data.append(smatrix.transmission(lead_out, lead_in))
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
            if 'current_up' in self.finished_calculations.keys() and 'current_down' in self.finished_calculations.keys():
                return self.finished_calculations['current_up'], self.finished_calculations['current_down']
            e, cond_up, cond_down = self.getConductance(lead_in, lead_out)
            de = e[1] - e[0]
            mu_left = bias / 2.0 + avg_chem_pot
            mu_right = -bias / 2.0 + avg_chem_pot
            diff_fermi = vectorizedFermi(e, mu_left, kb_T) - vectorizedFermi(e, mu_right, kb_T)
            self.finished_calculations['current_up'] = de * np.sum(cond_up * diff_fermi)
            self.finished_calculations['current_down'] = de * np.sum(cond_down * diff_fermi)
            return self.finished_calculations['current_up'], self.finished_calculations['current_down']

        else:
            if 'current' in self.finished_calculations.keys():
                return self.finished_calculations['current']
            e, cond = self.getConductance(lead_in, lead_out)
            de = e[1] - e[0]
            mu_left = bias / 2.0 + avg_chem_pot
            mu_right = -bias / 2.0 + avg_chem_pot
            diff_fermi = vectorizedFermi(e, mu_left, kb_T) - vectorizedFermi(e, mu_right, kb_T)
            self.finished_calculations['current'] = de * np.sum(cond * diff_fermi)
            return self.finished_calculations['current']



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
            if 'momenta' in self.finished_calculations.keys() and 'energies_up' in self.finished_calculations.keys() and 'energies_down' in self.finished_calculations.keys():
                return self.finished_calculations['momenta'], self.finished_calculations['energies_up'], self.finished_calculations['energies_down']
            bands_up = kwant.physics.Bands(self.system_up.leads[lead_id])
            bands_down = kwant.physics.Bands(self.system_down.leads[lead_id])
            energies_up = [bands_up(k) for k in momenta]
            energies_down = [bands_down(k) for k in momenta]
            self.finished_calculations['momenta'] = momenta
            self.finished_calculations['energies_up'] = energies_up
            self.finished_calculations['energies_down'] = energies_down
            return momenta, energies_up, energies_down         
        else:
            if 'momenta' in self.finished_calculations.keys() and 'energies' in self.finished_calculations.keys():
                return self.finished_calculations['momenta'], self.finished_calculations['energies']
            bands = kwant.physics.Bands(self.system.leads[lead_id])
            energies = [bands(k) for k in momenta]
            self.finished_calculations['momenta'] = momenta
            self.finished_calculations['energies'] = energies
            self.bands = energies
            return momenta, energies

    def getEnergyRange(self):
        """
        Find the energy range that must be integrated over to calcuate the current.

        Returns
        -------
        A list of length 2 with the zeroth element being the minimum and the first element being the maximum.
        """

        # try to avoid a calculation and return what was computed before
        if 'energy_range' in self.finished_calculations.keys():
            return self.finished_calculations['energy_range']
        
        if self.spin_dep:
            mins = []
            maxs = []
            for i, l in enumerate(self.system_up.leads):
                m, e_up, e_down = self.getBandStructure(i)
                mins.append(np.min(np.array(e_up).flatten()))
                maxs.append(np.max(np.array(e_up).flatten()))
                mins.append(np.min(np.array(e_down).flatten()))
                maxs.append(np.max(np.array(e_down).flatten()))
                self.energy_range = [min(mins), max(maxs)]
                self.finished_calculations['energy_range'] = self.energy_range
            return self.energy_range
        else:
            mins = []
            maxs = []
            for i, l in enumerate(self.system.leads):
                m, e = self.getBandStructure(i)
                mins.append(np.min(np.array(e).flatten()))
                maxs.append(np.max(np.array(e).flatten()))
                self.energy_range = [min(mins), max(maxs)]
                self.finished_calculations['energy_range'] = self.energy_range
            return self.energy_range

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
            if 'energies_DOS' in self.finished_calculations.keys() and 'DOS_up' in self.finished_calculations.keys() and 'DOS_down' in self.finished_calculations.keys():
                return self.finished_calculations['energies_DOS'], self.finished_calculations['DOS_up'], self.finished_calculations['DOS_down']
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
            self.finished_calculations['energies_DOS'] = es
            self.finished_calculations['DOS_up'] = DOS_up
            self.finished_calculations['DOS_down'] = DOS_down
            return es, DOS_up, DOS_down
        else:
            if 'energies_DOS' in self.finished_calculations.keys() and 'DOS' in self.finished_calculations.keys():
                return self.finished_calculations['energies_DOS'], self.finished_calculations['DOS']
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
            self.finished_calculations['energies_DOS'] = es
            self.finished_calculations['DOS'] = DOS
            return es, DOS

    def getValleyPolarizedConductance(self, energy, lead_start=0, lead_end=1, K_prime_range=(-np.inf, -1e-8), K_range=(0, np.inf), velocities='out_going'):
        """
        Get the valley-polarized conductances for a given energy between two leads. Note: This function only makes sense when
        the bandstructure has two valleys in it. An example is zig-zag edged graphene nanoribbons.

        Parameters
        ----------
        energy : A value of energy.
        lead_start : An integer of the lead where electrons are injected.
        lead_end : An integer of the lead where electrons are transmitting through.
        K_prime_range : A tuple of length 2 which defines the range where K' would be the polarization. Default: (-np.inf, -1e-8)
        K_range : A tuple of length 2 which defines the range where K would be the polarization. Default (0, np.inf)
        velocities : If 'out_going', we only consider velocities <= 0. If 'in_coming', velocities > 0. Default: 'out_going'

        Returns
        -------
        A tuple of length 2. First element is for K', second for K.
        """

        if self.spin_dep:
            logger.error('You are trying to compute the valley conductances for Spin-Polarized calculations. This is currently not supported.')
            exit(-1)

        keys = [
            'k_prime_conductance_' + str(lead_start) + '_' + str(lead_end) + '_' + str(energy),
            'k_conductance_' + str(lead_start) + '_' + str(lead_end) + '_' + str(energy)
        ]


        if set(keys).issubset(self.finished_calculations.keys()):
            return self.finished_calculations[keys[0]], self.finished_calculations[keys[1]]

        smatrix = kwant.smatrix(self.system, energy)
        if velocities == 'out_going':
            positives = np.where(smatrix.lead_info[lead_start].velocities <= 0)[0]
        elif velocities == 'in_coming':
            positives = np.where(smatrix.lead_info[lead_start].velocities > 0)[0]
        else:
            logger.error("You have defined the direction of the velocities wrong. It is either 'out_going' or 'in_coming'.")

        momentas = smatrix.lead_info[lead_start].momenta[positives]
        K_prime_indices = np.where(np.logical_and(momentas >= K_prime_range[0], momentas <= K_prime_range[1]))[0]
        K_prime_indices = np.where(momentas < 0)
        K_indices = np.where(np.logical_and(momentas >= K_range[0], momentas <= K_range[1]))[0]
        submatrix = smatrix.submatrix(lead_end, lead_start)
        K_prime_T = np.sum(np.absolute(submatrix[:, K_prime_indices])**2) 
        K_T = np.sum(np.absolute(submatrix[:, K_indices])**2)
        self.finished_calculations[keys[0]] = K_prime_T
        self.finished_calculations[keys[1]] = K_T
        return (K_prime_T, K_T)

    def getValleyPolarizedCurrent(self, lead_start=0, lead_end=1, K_prime_range=(-np.inf, -1e-8), K_range=(0, np.inf), velocities='out_going', avg_chem_pot=0.0):
        """
        Get the valley-polarized currents between two leads. Note: This function only makes sense when
        the bandstructure has two valleys in it. An example is zig-zag edged graphene nanoribbons.

        Parameters
        ----------
        lead_start : An integer of the lead where electrons are injected.
        lead_end : An integer of the lead where electrons are transmitting through.
        K_prime_range : A tuple of length 2 which defines the range where K' would be the polarization. Default: (-np.inf, -1e-8)
        K_range : A tuple of length 2 which defines the range where K would be the polarization. Default (0, np.inf)
        velocities : If 'out_going', we only consider velocities <= 0. If 'in_coming', velocities > 0. Default: 'out_going'
        avg_chem_pot : The average of the chemical potentials between two leads. Default: 1.0.

        Returns
        -------
        A tuple of length 2. First element is for K', second for K.
        """

        pid = os.getpid()
        bias = self.parser.getBias()
        kb_T = self.parser.getKBT()

        if self.spin_dep:
            logger.error('Cannot calculate valley dependent currents for spin-dependent systems.')
            exit(-1)

        keys = [
            'k_prime_current_' + str(lead_start) + '_' + str(lead_end),
            'k_current_' + str(lead_start) + '_' + str(lead_end)
        ]


        if set(keys).issubset(self.finished_calculations.keys()):
            return self.finished_calculations[keys[0]], self.finished_calculations[keys[1]]


        energy_range = self.getEnergyRange()
        energies = np.linspace(energy_range[0], energy_range[1], self.grid_size)
        

        KPs = []
        Ks = []
        start = time.time()
        for e in energies:
            vals = self.getValleyPolarizedConductance(e, lead_start, lead_end, K_prime_range, K_range, velocities)
            KPs.append(vals[0])
            Ks.append(vals[1])
        
        logger.info('Conductance calculation summary: pid - %i, leads - (%i, %i), structure - %i, time - %0.2f min' % (pid, lead_end, lead_start, self.identifier, float(time.time() - start) / 60.))

        KPs = np.array(KPs)
        Ks = np.array(Ks)

        de = energies[1] - energies[0]
        mu_left = bias / 2.0 + avg_chem_pot
        mu_right = -bias / 2.0 + avg_chem_pot
        diff_fermi = vectorizedFermi(energies, mu_left, kb_T) - vectorizedFermi(energies, mu_right, kb_T)
        self.finished_calculations[keys[0]] = de * np.sum(KPs * diff_fermi)
        self.finished_calculations[keys[1]] = de * np.sum(Ks * diff_fermi)

        return self.finished_calculations[keys[0]], self.finished_calculations[keys[1]]


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
        return self.parser.getPNJunction()['points']

