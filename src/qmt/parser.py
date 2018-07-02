#!/usr/bin/env python

import yaml
import coloredlogs, verboselogs
from functools import partial
from .shapes import *
from .masks import *

# create logger for parser
coloredlogs.install(level='INFO')
logger = verboselogs.VerboseLogger('QMT - Parser')


class Parser:
    """
    Class for reading in the YAML file: 'input.yaml', and parsing it such that methods return convenient objects.
    
    Steps:

    1. Open the file 'input.yaml'.
    2. Parse the Junction section of the file.
    3. Parse the GA section of the file.
    """
    def __init__(self):
        try:
            self.config = yaml.load(open('input.yaml', 'r'))
        except:
            logger.error('Could not parse the configuration file: "input.yaml".')

        # all of the variables set in self.parseModel
        self.body = None
        self.device = None
        self.mask = None
        self.leads = None
        self.parseModel()
        

        self.n_structures = None
        self.n_iterations = None
        self.parseGA()


    def parseModel(self):
        """
        Function that parses the Junction part of the input file.
        """
        junction_shapes = self.config['System']['Junction']
        if len(junction_shapes) == 0:
            logger.error('You have no shapes defined in junction!')
            exit(-1)
        else:
            shapes = []
            hoppings = []
            offsets = []
            pots = []
            phis = []
            for js in junction_shapes:
                if 'id' in js and js['id'] == 'body':
                    self.body = partial(whatShape(js['shape']), **js['args'])
                shapes.append(partial(whatShape(js['shape']), **js['args']))
                hoppings.append(js['hopping'])
                offsets.append(js['offset'])
                pots.append(js['potential'])
                phis.append(js['phi'])

            self.device =   {
                                'shapes': shapes,
                                'hoppings': hoppings,
                                'offsets': offsets,
                                'potentials': pots,
                                'phis': phis,
                                'body': self.body
                            }

        junction_masks = self.config['System']['Masks']
        if junction_masks is not None:
            masks = []
            for jm in junction_masks:
                masks.append(partial(whatMask(jm['name']), **jm['args']))

            self.mask = partial(multiMasks, masks)

        junction_leads = self.config['System']['Leads']
        if len(junction_leads) == 0:
            logger.error('You have not defined any leads!')
            exit(-1)
        else:
            pass

    def parseGA(self):
        """
        Function that parses the GA section of the input file.
        """
        self.n_structures = self.config['GA']['n_structures']
        self.n_iterations = self.config['GA']['n_iterations']

    def getNStructures(self):
        """
        Returns
        -------
        The number of GA structures.
        """
        return self.n_structures

    def getNIterations(self):
        """
        Returns
        -------
        The number of GA iterations.
        """
        return self.n_iterations

    def getDevice(self):
        """
        Returns
        -------
        The device which contains all the information needed to constuct it with Kwant.
        """
        return self.device

    def getMaskFunction(self):
        """
        Returns
        -------
        The mask function to be applied onto the body of the junction.
        """
        return self.mask

    def getLeads(self):
        """
        Returns
        -------
        The configuration of the leads.
        """
        return self.config['System']['Leads']

    def getLatticeType(self):
        """
        Returns
        -------
        The lattice type.
        """
        return self.config['System']['lattice_type']

    def getLatticeConstant(self):
        """
        Returns
        -------
        The lattice constant.
        """
        return self.config['System']['lattice_constant']

    def getLatticeVectors(self):
        """
        Returns
        -------
        The lattice vectors.
        """
        return self.config['System']['lattice_vectors']

    def getLatticeBasis(self):
        """
        Returns
        -------
        The lattice basis atoms.
        """
        return self.config['System']['lattice_basis']

    def getNumOrbitals(self):
        """
        Returns
        -------
        The number of orbitals on each site.
        """
        return self.config['System']['n_orbitals']

    def spinDependent(self):
        """
        Returns
        -------
        True if we want to do spin-dependent calculations, False if not.
        """
        return self.config['System']['spin_dependent']

    def getBias(self):
        """
        Returns
        -------
        The bias parameter.
        """
        return self.config['System']['bias']

    def getKBT(self):
        """
        Returns
        -------
        The Boltzmann constant times the temperature.
        """

