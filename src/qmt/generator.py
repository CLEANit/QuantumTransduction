from .system import Structure
from .helper import getFromDict, setInDict
from .system import Structure
import coloredlogs, verboselogs
import copy
import numpy as np
import random
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

    def checkAndUpdate(self, new_config, gene, val, new_val):
        # if we modify the body, we must update the channels
        clean_generation = True
        if 'body' in gene['path'] and 'xcoords' in gene['path']:
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
            
        if 'body' in gene['path'] and 'ycoords' in gene['path']:
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
        if 'channels' in gene['path']:
            # index of channel / lead
            i = gene['path'][3]
            channels = new_config['System']['Junction']['channels']

            # ctm - channel to modify
            ctm = new_config['System']['Junction']['channels'][i]
            # ltm - lead to modify
            ltm = new_config['System']['Leads'][gene['path'][3]]
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
        return new_config, clean_generation

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
            new_parser = copy.deepcopy(self.parser)
            old_config = self.parser.getConfig()
            new_config = new_parser.getConfig()
            for gene in self.parser.getGenes():
                val = getFromDict(old_config, gene['path'])
                new_val = copy.copy(val)
                if type(val) == list:
                    l = np.random.uniform(gene['range'][0], gene['range'][1])
                    s = np.random.uniform(gene['shift'][0], gene['shift'][1])
                    new_val[0] = - l / 2 + s
                    new_val[1] =   l / 2 + s
                elif type(val) == float:
                    new_val = np.random.uniform(gene['range'][0], gene['range'][1])
                setInDict(new_config, gene['path'], new_val)
                new_config, clean_generation = self.checkAndUpdate(new_config, gene, val, new_val)
                
        new_parser.updateConfig(new_config)

        return new_parser

    def mutate(self, structure):
        """
        Mutate the structures gene in some way.

        Parameters
        ----------
        A Structure class.

        Returns
        -------
        A new modified Structure class.
        
        """

        old_config = structure.parser.getConfig()
        new_parser = copy.deepcopy(structure.parser)
        new_config = new_parser.getConfig()
        # the gene we are modifying
        gene = random.choice(old_config['GA']['Genes'])
        
        # follow similar steps as when we generate a new structure
        clean_generation = False

        while not clean_generation:
            val = getFromDict(old_config, gene['path'])
            new_val = copy.copy(val)
            if type(val) == list:
                l = np.random.uniform(gene['range'][0], gene['range'][1])
                s = np.random.uniform(gene['shift'][0], gene['shift'][1])
                new_val[0] = - l / 2 + s
                new_val[1] =   l / 2 + s
            elif type(val) == float:
                new_val = np.random.uniform(gene['range'][0], gene['range'][1])
            setInDict(new_config, gene['path'], new_val)
            new_config, clean_generation = self.checkAndUpdate(new_config, gene, val, new_val)
                            
        new_parser.updateConfig(new_config)
        return Structure(new_parser)

    def mutateAll(self, structures, pool=None):
        """
        Mutate all structures. Possibly in parallel.

        Parameters
        ----------
        structures : A list of structure classes that the mutate function will be called on.
        pool : A process pool so the mutations can be done in parallel. Default is None.

        Returns
        -------
        A mutated structure class.
        """
        if pool is None:
            return [self.mutate(s) for s in structures]
        else:
            return pool.map(self.mutate, structures)


    def generateAll(self):
        """
        Generates all of the structures specified in the input file.

        Returns
        -------
        A list of Parser classes that can be handed to Structure classes.

        """
        return [self.generate() for i in range(self.parser.getNStructures())]
