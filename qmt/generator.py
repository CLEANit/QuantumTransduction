from .system import Structure
from .helper import getFromDict, setInDict
from .system import Structure
import coloredlogs, verboselogs
import copy
import numpy as np
import random
from sklearn.neural_network import MLPRegressor


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
        self.n_generated = 0

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


    def generate(self, seed=None, thread_num=None):
        """
        Generate a random structure based on the genes given in the output.

        Parameters
        ----------
        seed : A random seed for setting the weights.

        Returns
        -------
        A Parser class that can be passed to the structure class.
        """
        # we want to make sure our generated structure is good

        ann = self.parser.getGAParameters()['ann']
        random.seed(seed)
        np.random.seed(seed)

        if ann:
            clean_generation = False

            while not clean_generation:
                new_parser = copy.deepcopy(self.parser)

                ann_params = self.parser.getAnnParameters()
                ga_params = self.parser.getGAParameters()

                ann = MLPRegressor(
                        hidden_layer_sizes=tuple(ann_params['neurons']) + (len(self.parser.getGenes()),),
                        activation=ann_params['activation']
                    )
                layers = [ann_params['neurons'][0]] + ann_params['neurons'] + [len(self.parser.getGenes())]
                input_vec = np.ones((1, ann_params['neurons'][0]))
                output_vec = np.empty((1, len(self.parser.getGenes())))
                ann._random_state = np.random.RandomState(seed)

                ann._initialize(output_vec, layers)
                ann.out_activation_ = ann_params['activation']

                new_parser.ann = ann

                outputs = new_parser.ann.predict(input_vec)

                old_config = self.parser.getConfig()
                new_config = new_parser.getConfig()
                for gene, output in zip(self.parser.getGenes(), outputs[0]):
                    val = getFromDict(old_config, gene['path'])
                    new_val = (gene['range'][1] - gene['range'][0]) * output + gene['range'][0]
                    setInDict(new_config, gene['path'], new_val)
                    new_config, clean_generation = self.checkAndUpdate(new_config, gene, val, new_val)
                    
            new_parser.updateConfig(new_config)

            identifier = self.n_generated + thread_num
            history = [self.n_generated]
            s = Structure(new_parser, identifier, history)
            return s
        else:
            clean_generation = False

            while not clean_generation:
                new_parser = copy.deepcopy(self.parser)
                old_config = self.parser.getConfig()
                new_config = new_parser.getConfig()
                for gene in self.parser.getGenes():
                    val = getFromDict(old_config, gene['path'])
                    new_val = (gene['range'][1] - gene['range'][0]) * np.random.uniform() + gene['range'][0]
                    setInDict(new_config, gene['path'], new_val)
                    new_config, clean_generation = self.checkAndUpdate(new_config, gene, val, new_val)

            new_parser.updateConfig(new_config)

            identifier = self.n_generated + thread_num
            s = Structure(new_parser, identifier, [])
            return s

    def crossOver(self, pair_of_structures, seed=None, thread_num=None):
        """
        Crossover the structures genes randomly according to the input parameters.

        Parameters
        ----------
        structure1 : A Structure class.
        structure2 : Another Structure class. 
        seed : A random seed to handle multithreading properly. Default: None.
        ann: Use an ANN to generate the genes. Default: True.

        Returns
        -------
        A new modified Structure class where the genes will be mixed from structure1 and structure2.
        
        """
        ann = self.parser.getGAParameters()['ann']

        if self.parser.getGenerator()['turn_on']:
            structure1, structure2 = pair_of_structures
            ga_params = self.parser.getGAParameters()
            generator_params = self.parser.getGenerator()
            random.seed(seed)
            np.random.seed(seed)        

            # follow similar steps as when we generate a new structure
            clean_generation = False

            while not clean_generation:
                old_config = structure1.parser.getConfig()
                new_parser = copy.deepcopy(structure1.parser)
                new_config = new_parser.getConfig()
                for layer in range(len(generator_params['neurons'])):
                    total_weights = new_parser.policy_mask.coefs_[layer].shape[0] * new_parser.policy_mask.coefs_[layer].shape[1]
                    indices_to_update = np.vstack((np.random.randint(0, new_parser.policy_mask.coefs_[layer].shape[0], size=int(total_weights * ga_params['random-step']['fraction'])), np.random.randint(0, new_parser.policy_mask.coefs_[layer].shape[1], size=int(total_weights * ga_params['random-step']['fraction'])))).T

                    new_parser.policy_mask.coefs_[layer][indices_to_update[:,0], indices_to_update[:,1]] = structure2.parser.policy_mask.coefs_[layer][indices_to_update[:,0], indices_to_update[:,1]] 
                    
            new_parser.updateConfig(new_config)
            
            identifier = self.n_generated + thread_num
            # history = []
            # for h1, h2 in zip(structure1.history, structure2.history):
            #     history.append([h1, h2])
            # history.append([structure1.identifier, structure2.identifier])
            s = Structure(new_parser, identifier, [structure1.identifier, structure2.identifier])
            return s

        if ann:
            structure1, structure2 = pair_of_structures
            ann_params = self.parser.getAnnParameters()
            ga_params = self.parser.getGAParameters()
            random.seed(seed)
            np.random.seed(seed)        

            input_vec = np.ones((1, ann_params['neurons'][0]))

            # follow similar steps as when we generate a new structure
            clean_generation = False

            while not clean_generation:
                old_config = structure1.parser.getConfig()
                new_parser = copy.deepcopy(structure1.parser)
                new_config = new_parser.getConfig()

                for layer in range(len(ann_params['neurons']) + 1):
                    total_weights = new_parser.ann.coefs_[layer].shape[0] * new_parser.ann.coefs_[layer].shape[1]
                    indices_to_update = np.vstack((np.random.randint(0, new_parser.ann.coefs_[layer].shape[0], size=int(total_weights * ga_params['random-step']['fraction'])), np.random.randint(0, new_parser.ann.coefs_[layer].shape[1], size=int(total_weights * ga_params['random-step']['fraction'])))).T

                    new_parser.ann.coefs_[layer][indices_to_update[:,0], indices_to_update[:,1]] = structure2.parser.ann.coefs_[layer][indices_to_update[:,0], indices_to_update[:,1]] 

                outputs = new_parser.ann.predict(input_vec)[0]
            
                for gene, output in zip(self.parser.getGenes(), outputs):
                    val = getFromDict(old_config, gene['path'])
                    new_val = (gene['range'][1] - gene['range'][0]) * output + gene['range'][0]
                    setInDict(new_config, gene['path'], new_val)
                    new_config, clean_generation = self.checkAndUpdate(new_config, gene, val, new_val)
                    
            new_parser.updateConfig(new_config)
            
            identifier = self.n_generated + thread_num
            # history = []
            # for h1, h2 in zip(structure1.history, structure2.history):
            #     history.append([h1, h2])
            # history.append([structure1.identifier, structure2.identifier])
            s = Structure(new_parser, identifier, [structure1.identifier, structure2.identifier])

            return s

        else:
            structure1, structure2 = pair_of_structures
            ga_params = self.parser.getGAParameters()
            random.seed(seed)
            np.random.seed(seed)        

            # follow similar steps as when we generate a new structure
            clean_generation = False

            while not clean_generation:
                old_config = structure1.parser.getConfig()
                new_parser = copy.deepcopy(structure1.parser)
                new_config = new_parser.getConfig()
            
                outputs = list(zip(structure1.parser.getPNJunction()['points'], structure2.parser.getPNJunction()['points']))
                for gene, output in zip(self.parser.getGenes(), outputs):
                    if np.random.uniform() < ga_params['crossing-fraction']:
                        val = getFromDict(old_config, gene['path'])
                        index = gene['path'][-2:]
                        new_val = np.mean([outputs[index[0]][0][index[1]], outputs[index[0]][1][index[1]]])
                        setInDict(new_config, gene['path'], new_val)
                        new_config, clean_generation = self.checkAndUpdate(new_config, gene, val, new_val)
                    
            new_parser.updateConfig(new_config)
                                
            identifier = self.n_generated + thread_num
            # history = []
            # for h1, h2 in zip(structure1.history, structure2.history):
            #     history.append([h1, h2])
            # history.append([structure1.identifier, structure2.identifier])
            s = Structure(new_parser, identifier, [structure1.identifier, structure2.identifier])

            return s



    def mutate(self, structure, seed=None):
        """
        Mutate the structures gene in some way.

        Parameters
        ----------
        A Structure class.

        Returns
        -------
        A new modified Structure class.
        
        """
        ann = self.parser.getGAParameters()['ann']

        if self.parser.getGenerator()['turn_on']:
            generator_params = self.parser.getGenerator()
            ga_params = self.parser.getGAParameters()
            if ga_params['random-step']['fraction'] == 0.:
                return structure

            random.seed(seed)
            np.random.seed(seed)        

            # follow similar steps as when we generate a new structure
            clean_generation = False

            while not clean_generation:
                old_config = structure.parser.getConfig()
                new_parser = copy.deepcopy(structure.parser)
                new_config = new_parser.getConfig()

                for layer in range(len(generator_params['neurons']) + 1):
                    total_weights = new_parser.policy_mask.coefs_[layer].shape[0] * new_parser.policy_mask.coefs_[layer].shape[1]
                    indices_to_update = np.vstack((np.random.randint(0, new_parser.policy_mask.coefs_[layer].shape[0], size=int(total_weights * ga_params['random-step']['fraction'])), np.random.randint(0, new_parser.policy_mask.coefs_[layer].shape[1], size=int(total_weights * ga_params['random-step']['fraction'])))).T
                    new_parser.policy_mask.coefs_[layer][indices_to_update[:,0], indices_to_update[:,1]] += ga_params['random-step']['max-update-rate'] * np.random.uniform(size=indices_to_update.shape[0])
                    
            new_parser.updateConfig(new_config)
            
            # identifier = self.n_generated
            # # history = copy.copy(structure.history)
            # # history.append(structure.identifier)
            s = Structure(new_parser, structure.identifier, structure.parents)
            # self.n_generated += 1

            return s

        if ann:
            ann_params = self.parser.getAnnParameters()
            ga_params = self.parser.getGAParameters()
            if ga_params['random-step']['fraction'] == 0.:
                return structure

            random.seed(seed)
            np.random.seed(seed)        

            input_vec = np.ones((1, ann_params['neurons'][0]))

            # follow similar steps as when we generate a new structure
            clean_generation = False

            while not clean_generation:
                old_config = structure.parser.getConfig()
                new_parser = copy.deepcopy(structure.parser)
                new_config = new_parser.getConfig()

                for layer in range(len(ann_params['neurons']) + 1):
                    total_weights = new_parser.ann.coefs_[layer].shape[0] * new_parser.ann.coefs_[layer].shape[1]
                    indices_to_update = np.vstack((np.random.randint(0, new_parser.ann.coefs_[layer].shape[0], size=int(total_weights * ga_params['random-step']['fraction'])), np.random.randint(0, new_parser.ann.coefs_[layer].shape[1], size=int(total_weights * ga_params['random-step']['fraction'])))).T
                    new_parser.ann.coefs_[layer][indices_to_update[:,0], indices_to_update[:,1]] += ga_params['random-step']['max-update-rate'] * np.random.uniform(size=indices_to_update.shape[0])

                outputs = new_parser.ann.predict(input_vec)[0]
            
                for gene, output in zip(self.parser.getGenes(), outputs):
                    val = getFromDict(old_config, gene['path'])
                    new_val = (gene['range'][1] - gene['range'][0]) * output + gene['range'][0]
                    setInDict(new_config, gene['path'], new_val)
                    new_config, clean_generation = self.checkAndUpdate(new_config, gene, val, new_val)
                    
            new_parser.updateConfig(new_config)
            
            # identifier = self.n_generated
            # # history = copy.copy(structure.history)
            # # history.append(structure.identifier)
            s = Structure(new_parser, structure.identifier, structure.parents)
            # self.n_generated += 1

            return s
        else:
            ga_params = self.parser.getGAParameters()
            if ga_params['random-step']['fraction'] == 0.:
                return structure

            random.seed(seed)
            np.random.seed(seed)        

            # follow similar steps as when we generate a new structure
            clean_generation = False

            while not clean_generation:
                old_config = structure.parser.getConfig()
                new_parser = copy.deepcopy(structure.parser)
                new_config = new_parser.getConfig()

                outputs = structure.parser.getPNJunction()['points']
            
                for gene, output in zip(self.parser.getGenes(), outputs):
                    if np.random.uniform() < ga_params['random-step']['fraction']:
                        val = getFromDict(old_config, gene['path'])
                        new_val = val + ga_params['random-step']['max-update-rate'] * np.random.uniform()

                        # check to make sure we are in the range, if not, wrap the value 
                        if new_val < gene['range'][0]:
                            new_val += (gene['range'][1] - gene['range'][0])
                        elif new_val > gene['range'][1]:
                            new_val -= (gene['range'][1] - gene['range'][0])
                        else:
                            # we're within the defined range
                            pass
                        setInDict(new_config, gene['path'], new_val)
                        new_config, clean_generation = self.checkAndUpdate(new_config, gene, val, new_val)
                    
            new_parser.updateConfig(new_config)
                                
            # identifier = self.n_generated
            # # history = copy.copy(structure.history)
            # # history.append(structure.identifier)
            s = Structure(new_parser, structure.identifier, structure.parents)
            # self.n_generated += 1

            return s

    def mutateAllWeights(self, structure, seed=None):
        """
        Mutate the structures gene in some way.

        Parameters
        ----------
        A Structure class.

        Returns
        -------
        A new modified Structure class.
        
        """
        ann_params = self.parser.getAnnParameters()
        if ann_params['random-step']['fraction'] == 0.:
            return structure

        random.seed(seed)
        np.random.seed(seed)        

        input_vec = np.ones((1, ann_params['neurons'][0]))

        # follow similar steps as when we generate a new structure
        clean_generation = False

        while not clean_generation:
            old_config = structure.parser.getConfig()
            new_parser = copy.deepcopy(structure.parser)
            new_config = new_parser.getConfig()

            for layer in range(len(ann_params['neurons']) + 1):
                new_parser.ann.coefs_[layer] += ann_params['random-step']['max-update-rate'] * np.random.normal(size=new_parser.ann.coefs_[layer].shape)

            outputs = new_parser.ann.predict(input_vec)[0]
        
            for gene, output in zip(self.parser.getGenes(), outputs):
                val = getFromDict(old_config, gene['path'])
                new_val = (gene['range'][1] - gene['range'][0]) * output + gene['range'][0]
                setInDict(new_config, gene['path'], new_val)
                new_config, clean_generation = self.checkAndUpdate(new_config, gene, val, new_val)
                
        new_parser.updateConfig(new_config)
                            
        return Structure(new_parser)

    def mutateAll(self, structures, pool=None, seeds=None):
        """
        Mutate all structures. Possibly in parallel.

        Parameters
        ----------
        structures : A list of structure classes that the mutate function will be called on.
        pool : A process pool so the mutations can be done in parallel. Default is None.
        seeds : List of seeds to properly handle multithreading.

        Returns
        -------
        A list of mutated structure classes.
        """
        if pool is None:
            return [self.mutate(s) for s in structures]
        else:
            return pool.map(self.mutate, structures, seeds)

    def crossOverAll(self, pairs_of_structures, pool=None, seeds=None):
        """
        Crossover all pairs of structures.

        Parameters
        ----------
        pairs_of_structures : List of pairs of structures.
        pool : A process pool so the crossovers can be done in parallel. Default is None.
        seeds : List of seeds to properly handle multithreading.

        Returns
        -------
        A list of new structure classes with genes crossed over from the pairs_of_structures.

        """
        if pool is None:
            css =  [self.crossOver(s) for s in pairs_of_structures]
        else:
            css =  pool.map(self.crossOver, pairs_of_structures, seeds, range(len(pairs_of_structures)))
        self.n_generated += len(pairs_of_structures)
        return css

    def generateAll(self, pool=None, seeds=None):
        """
        Generates all of the structures specified in the input file.

        Returns
        -------
        A list of Parser classes that can be handed to Structure classes.
        """
        if pool is None:
            ss = [self.generate() for i in range(self.parser.getNStructures())]
        else:
            ss =  pool.map(self.generate, seeds, range(self.parser.getNStructures()))
        self.n_generated += self.parser.getNStructures()
        return ss