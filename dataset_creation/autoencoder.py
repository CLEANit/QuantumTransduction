#!/usr/bin/env python


from src.qmt.system import Structure
from src.qmt.generator import Generator
from src.qmt.parser import Parser
from src.qmt.timer import Timer
import dill
import numpy as np

import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool

import coloredlogs, verboselogs
import copy
import matplotlib.pyplot as plt
import scipy
import progressbar
# create logger
coloredlogs.install(level='INFO')
bar = progressbar.ProgressBar()
logger = verboselogs.VerboseLogger('qmt::autoecoder ')


def convertToImages(structures):
    ims = []
    for s in structures:
        im = s.getBinaryRepresentation()
        ims.append(im)
    return np.array(ims)

def writeAdjacencyMatrixAndFeatures(structures):
    Hs = []
    features = []
    for s in bar(structures):
        H = s.system.hamiltonian_submatrix(sparse=True).astype(np.float32)
        feature = H.diagonal()
        H.setdiag(np.zeros(H.shape[0]))
        H /= s.parser.config['System']['Junction']['body']['hopping']
        Hs.append(H)
        features.append(feature)
    Hs = scipy.sparse.block_diag(Hs)
    features = scipy.sparse.block_diag(features).T
    scipy.sparse.save_npz('systems.npz', Hs)
    scipy.sparse.save_npz('features.npz', features)

def main():
    total_timer = Timer()
    short_timer = Timer()
    total_timer.start()
    pool = Pool()

    parser = Parser()
    g = Generator(parser)
    
    short_timer.start()
    try:
        structures = dill.load(open('structures.dill', 'rb'))
    except:
        logger.warning('Could not find structures file. Will start fresh.')
        parsers = g.generateAll()
        structures = pool.map(Structure, parsers)

    logger.success('Structures generated/loaded. Elapsed time: %s' % (short_timer.stop()))


    # writeAdjacencyMatrixAndFeatures(structures)
    ims = convertToImages(structures)
    np.save('structures.npy', ims)

    # logger.info('Serializing structures.')
    # short_timer.start()
    # dill.dump(structures, open('structures.dill', 'wb'))

    logger.success('Structures serialized. Elapsed time: %s' % (total_timer.stop()))
        
if __name__ == '__main__':
    main()
