#!/usr/bin/env python

from src.model import Model
from src.shapes import *
import numpy as np
from matplotlib import pyplot
import kwant

from functools import partial

import coloredlogs, verboselogs
# create logger
coloredlogs.install(level='DEBUG')
logger = verboselogs.VerboseLogger('QMtransport')

pot = 0.1

def potential(site):
    (x, y) = site.pos
    d = y * cos_30 + x * sin_30
    return pot * np.tanh(d * 0.5)

def main():
    l = 200
    w = 100
    body = partial(rectangle, -l, l, -w, w)
    lc = partial(rectangle, -l - w/2, -l + 1, -w/4, w/4)
    ruc = partial(rectangle, l - 1, l + w/2, w/4, 3 * w / 4)
    rlc = partial(rectangle, l - 1, l + w/2, -3 * w / 4, -w/4)

    device = partial(rectDevice, body, lc, ruc, rlc)

    lead_shapes = [partial(rectangle, -50, 50, -w/4, w/4), partial(rectangle, -50, 50, w/4, 3 * w / 4), partial(rectangle, -50, 50, -3 * w / 4, -w/4)]
    m = Model(  
                logger,
                device,
                potential,
                lead_shapes,
                [(-1,0), (1,0), (1, 0)], 
                [(0,0), (0, w / 4), (0, -w/4)],
                [-pot, pot, pot],
                shape_offset=(0, 0)
             )

    syst_fig = m.visualizeSystem()

    logger.info('Number of sites: %i' % (m.getNSites()))
    
    m.finalize()
    m.plotCurrent(0)

if __name__ == '__main__':
    main()