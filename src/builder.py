#!/usr/bin/env python

import kwant
import numpy as np

def parentStructure(args):
    body, shape, potential, lattice_const, norbs = args
    lattice = kwant.lattice.honeycomb(lattice_const, norbs=norbs)
    system = kwant.Builder()
    system[lattice.shape(shape, (0,0))] = potential
    return system


def applyMask(system, body, mask):
    tags = []
    positions = []
    sites = []
    for s, v in system.site_value_pairs():
        # if the site is in the body
        if body(s.pos):
            tags.append(s.tag)
            positions.append(s.pos)
            sites.append(s)
        # print (s.tag)
    tags = np.array(tags)
    positions = np.array(positions)
    min_tag_sx = np.min(tags[:,0])
    min_tag_sy = np.min(tags[:,1])
    min_pos_sx = np.min(positions[:,0])
    min_pos_sy = np.min(positions[:,1])
    max_pos_sx = np.max(positions[:,0])
    max_pos_sy = np.max(positions[:,1])

    tag_length = np.max(tags[:,0]) - min_tag_sx
    tag_width = np.max(tags[:,1]) - min_tag_sy

    tags[:,0] += np.abs(min_tag_sx)
    tags[:,1] += np.abs(min_tag_sy)
    positions[:, 0] += np.abs(min_pos_sx)
    positions[:, 1] += np.abs(min_pos_sy)

    removed_tags = np.argwhere(mask((tag_length, tag_width), (min_pos_sx, min_pos_sy), (max_pos_sx, max_pos_sy), positions) == 0).astype(int)
    removed_tags = removed_tags.reshape(removed_tags.shape[0]).astype(int)
    for elem in removed_tags:
        del system[sites[int(elem)]]
    system.eradicate_dangling()
    return system
