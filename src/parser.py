#!/usr/bin/env python

import yaml
import coloredlogs, verboselogs
from functools import partial
from .shapes import *

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
        self.parseModel()
        

        self.n_structures = None
        self.parseGA()


    def parseModel(self):
        junction_shapes = self.config['Model']['Junction']
        if len(junction_shapes) == 0:
            logger.error('You have no shapes defined in junction!')
            exit(-1)
        else:
            shapes = []
            for js in junction_shapes:
                if 'id' in js and js['id'] == 'body':
                    self.body = partial(whatShape(js['shape']), **js['args'])
                shapes.append(partial(whatShape(js['shape']), **js['args']))

            self.device = partial(nBodyDevice, shapes)

    def parseGA(self):
        self.n_structures = self.config['GA']['n_structures']

    def getNStructures(self):
        return self.n_structures

    def getDevice(self):
        return self.body, self.device