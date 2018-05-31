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
            logger.error('Could not parser configuration file: "input.yaml".')

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
        junction_shapes = self.config['Model']['Junction']
        if len(junction_shapes) == 0:
            logger.error('You have no shapes defined in junction!')
            exit(-1)
        else:
            shapes = []
            hoppings = []
            offsets = []
            pots = []
            for js in junction_shapes:
                if 'id' in js and js['id'] == 'body':
                    self.body = partial(whatShape(js['shape']), **js['args'])
                shapes.append(partial(whatShape(js['shape']), **js['args']))
                hoppings.append(js['hopping'])
                offsets.append(js['offset'])
                pots.append(js['potential'])

            self.device =   {
                                'shapes': shapes,
                                'hoppings': hoppings,
                                'offsets': offsets,
                                'potentials': pots,
                                'body': self.body
                            }

        junction_masks = self.config['Model']['Masks']
        if junction_masks is not None:
            masks = []
            for jm in junction_masks:
                masks.append(partial(whatMask(jm['name']), **jm['args']))

            self.mask = partial(multiMasks, masks)

        junction_leads = self.config['Model']['Leads']
        if len(junction_leads) == 0:
            logger.error('You have not defined any leads!')
            exit(-1)
        else:
            self.leads = []
            for jl in junction_leads:
                lead = {}
                lead['shape'] = partial(whatShape(jl['shape']), **jl['shape_args'])
                lead['symmetry'] = jl['symmetry']
                lead['offset'] = jl['offset']
                lead['potential'] = jl['potential']
                lead['hopping'] = jl['hopping']
                self.leads.append(lead)


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
        return self.leads