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
        new_parser = copy.copy(self.parser)
        old_config = self.parser.getConfig()
        new_config = new_parser.getConfig()
        for elem in self.parser.getGenes():
            val = getFromDict(old_config, elem['path'])
            if type(val) == list:
                l = np.random.uniform(elem['range'][0], elem['range'][1])
                val[0] = - l / 2
                val[1] =   l / 2
            elif type(val) == float:
                pass
            setInDict(new_config, elem['path'], val)

            # if we modify the body, we must update the channels and leads
            if 'body' in elem['path'] and 'xcoords' in elem['path']:
                channels = old_config['System']['Junction']['channels']
                for c in channels:
                    if c['args']['xcoords'][0] < 0:
                        c['args']['shift'][0] = val[0]
                        c['offset'][0] = val[0]
                    else:
                        c['args']['shift'][0] = val[1]
                        c['offset'][0] = val[1]
                
                new_config['System']['Junction']['channels'] = channels

            if 'body' in elem['path'] and 'ycoords' in elem['path']:
                channels = old_config['System']['Junction']['channels']
                for c in channels:
                    if c['args']['ycoords'][0] < 0:
                        c['args']['shift'][1] = val[0]
                        c['offset'][1] = val[0]
                    else:
                        c['args']['shift'][1] = val[1]
                        c['offset'][1] = val[1]
                
                new_config['System']['Junction']['channels'] = channels
                print(channels)
        new_parser.resetConfig(new_config)
    def generateAll(self):
        return [self.generate() for i in range(self.parser.getNStructures())]