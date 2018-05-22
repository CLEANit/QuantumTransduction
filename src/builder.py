#!/usr/bin/env python

import kwant
import numpy as np
import random

def parentStructure(args):
    body, device, potential, lattice_const, norbs = args
    lattice = kwant.lattice.honeycomb(lattice_const, norbs=norbs)
    system = kwant.Builder()
    for shape, hopping, offset in zip(device['shapes'], device['hoppings'], device['offsets']):
        system[lattice.shape(shape, offset)] = potential
        neighbors = lattice.neighbors()
        system[neighbors] = hopping
    system.neighbors = lattice.neighbors()
    system.lattice = lattice
    return system


def applyMask(args):
    system, body, mask = args
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

    masc, info = mask((tag_length, tag_width), (min_pos_sx, min_pos_sy), (max_pos_sx, max_pos_sy), positions)
    removed_tags = np.argwhere(masc == 0).astype(int)
    removed_tags = removed_tags.reshape(removed_tags.shape[0]).astype(int)
    for elem in removed_tags:
        del system[sites[int(elem)]]
    system.mask_info = info
    return system

def attachLead(args):
    system, leads = args
    for l in leads:
        sym = kwant.TranslationalSymmetry(system.lattice.vec(l['symmetry']))
        lead = kwant.Builder(sym)
        lead[system.lattice.shape(l['shape'], l['offset'])] = l['potential']
        lead[system.neighbors] = l['hopping']
        system.attach_lead(lead)
    system.eradicate_dangling()
    return system
