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
        self.n_structures = self.config['GA']['n_structures']
        self.n_iterations = self.config['GA']['n_iterations']

    def getNStructures(self):
        return self.n_structures

    def getNIterations(self):
        return self.n_iterations

    def getDevice(self):
        return self.device

    def getMaskFunction(self):
        return self.mask

    def getLeads(self):
        return self.config['System']['Leads']

    def getLatticeType(self):
        return self.config['System']['lattice_type']

    def getLatticeConstant(self):
        return self.config['System']['lattice_constant']

    def getLatticeVectors(self):
        return self.config['System']['lattice_vectors']

    def getLatticeBasis(self):
        return self.config['System']['lattice_basis']

    def getNumOrbitals(self):
        return self.config['System']['n_orbitals']

    def spinDependent(self):
        return self.config['System']['spin_dependent']

