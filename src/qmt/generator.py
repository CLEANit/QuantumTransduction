from .system import Structure
from .helper import getFromDict, setInDict

import coloredlogs, verboselogs
import copy
import numpy as np
# create logger
coloredlogs.install(level='INFO')
logger = verboselogs.VerboseLogger(' <-- QMT: Generator --> ')

class Generator:
    """
    This class generates structures based on the GA confuration. It takes a parser,
    tweaks it, and allows it to be ready to perform calculations on.

    Parameters
    ----------
    parser : YAML configuation.

    """
    def __init__(self, parser):
        self.parser = parser
        self.known_genes = ['bodyLength', 'bodyWidth', 'channelWidth', 'channelShift', 'magneticField']


    def generate(self):
        """
        Generate a random structure based on the genes given in the output.

        Returns
        -------
        A Parser class that can be passed to the structure class.
        """
        # we want to make sure our generated structure is good
        clean_generation = False

        while not clean_generation:
            # assume we start clean
            clean_generation = True

            new_parser = copy.deepcopy(self.parser)
            old_config = self.parser.getConfig()
            new_config = new_parser.getConfig()
            for elem in self.parser.getGenes():
                val = getFromDict(old_config, elem['path'])
                new_val = copy.copy(val)
                if type(val) == list:
                    l = np.random.uniform(elem['range'][0], elem['range'][1])
                    s = np.random.uniform(elem['shift'][0], elem['shift'][1])
                    new_val[0] = - l / 2 + s
                    new_val[1] =   l / 2 + s
                elif type(val) == float:
                    new_val = np.random.uniform(elem['range'][0], elem['range'][1])
                setInDict(new_config, elem['path'], new_val)

                # if we modify the body, we must update the channels
                if 'body' in elem['path'] and 'xcoords' in elem['path']:
                    channels = new_config['System']['Junction']['channels']
                    shift = new_val[1] - val[1]
                    for c in channels:
                        if c['direction'][0] < 0:
                            c['args']['shift'][0] -= shift
                            c['offset'][0] = -shift + (c['args']['xcoords'][0] + c['args']['xcoords'][1]) / 2
                        elif c['direction'][0] > 0:
                            c['args']['shift'][0] += shift
                            c['offset'][0] = shift + (c['args']['xcoords'][0] + c['args']['xcoords'][1]) / 2
                        else:
                            pass
                    
                if 'body' in elem['path'] and 'ycoords' in elem['path']:
                    channels = new_config['System']['Junction']['channels']
                    shift = new_val[1] - val[1]
                    for c in channels:
                        if c['direction'][1] < 0:
                            c['args']['shift'][1] -= shift
                            c['offset'][1] = -shift + (c['args']['ycoords'][0] + c['args']['ycoords'][1]) / 2
                        elif c['direction'][1] > 0:
                            c['args']['shift'][1] += shift
                            c['offset'][1] = shift + (c['args']['ycoords'][0] + c['args']['ycoords'][1]) / 2
                        else:
                            pass 

                # if we modify the channels, we must update the leads
                if 'channels' in elem['path']:
                    # index of channel / lead
                    i = elem['path'][3]
                    channels = new_config['System']['Junction']['channels']

                    # ctm - channel to modify
                    ctm = new_config['System']['Junction']['channels'][i]
                    # ltm - lead to modify
                    ltm = new_config['System']['Leads'][elem['path'][3]]
                    ltm['range'][0] = new_val[0]
                    ltm['range'][1] = new_val[1]

                    direction = ctm['direction']
                    if direction.index(0) == 0:
                        ctm['offset'][0] = (new_val[0] + new_val[1]) / 2
                        new_config['System']['Leads'][i]['offset'][0] = (new_val[0] + new_val[1]) / 2
                    elif direction.index(0) == 1:
                        new_config['System']['Leads'][i]['offset'][1] = (new_val[0] + new_val[1]) / 2
                        ctm['offset'][1] = (new_val[0] + new_val[1]) / 2

                    # check and see if we have overlapping channels

                    for j, ch in enumerate(channels):
                        if i != j and channels[i]['direction'] == channels[j]['direction']:
                            if channels[i]['direction'].index(0) == 0:
                                # check overlap of x
                                overlap1 = channels[i]['args']['xcoords'][0] >= channels[j]['args']['xcoords'][0] and channels[i]['args']['xcoords'][0] <= channels[j]['args']['xcoords'][1]
                                overlap2 = channels[i]['args']['xcoords'][1] >= channels[j]['args']['xcoords'][0] and channels[i]['args']['xcoords'][1] <= channels[j]['args']['xcoords'][1]
                                if overlap1 or overlap2:
                                    clean_generation = False
                            elif channels[i]['direction'].index(0) == 1:
                                # check overlap of y
                                overlap1 = channels[i]['args']['ycoords'][0] >= channels[j]['args']['ycoords'][0] and channels[i]['args']['ycoords'][0] <= channels[j]['args']['ycoords'][1]
                                overlap2 = channels[i]['args']['ycoords'][1] >= channels[j]['args']['ycoords'][0] and channels[i]['args']['ycoords'][1] <= channels[j]['args']['ycoords'][1]
                                if overlap1 or overlap2:
                                    clean_generation = False
                            else:
                                # pass for now
                                pass



        new_parser.updateConfig(new_config)

        return new_parser

    def generateAll(self):
        """
        Generates all of the structures specified in the input file.

        Returns
        -------
        A list of Parser classes that can be handed to Structure classes.

        """
        return [self.generate() for i in range(self.parser.getNStructures())]