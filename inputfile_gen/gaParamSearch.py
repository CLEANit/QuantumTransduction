#!/usr/bin/env python

import yaml
import os
import numpy as np

config = yaml.load(open('../input.yaml', 'r'))

os.mkdir('with_ann')
os.mkdir('without_ann')

n_tests = 10

x = np.linspace(0, 1, n_tests)

config['GA']['ann'] = False

for cf in x:
	for rsf in x:
		config['GA']['crossing-fraction'] = cf
		config['GA']['random-step']['fraction'] = rsf
		os.mkdir('without_ann/cross_fraction_' + str(cf) + '_random_step_fraction_' + str(rsf))
		yaml.dump(config, open('without_ann/cross_fraction_' + str(cf) + '_random_step_fraction_' + str(rsf) + '/input.yaml', 'w'))

config['GA']['ann'] = True

for cf in x:
	for rsf in x:
		config['GA']['crossing-fraction'] = cf
		config['GA']['random-step']['fraction'] = rsf
		os.mkdir('with_ann/cross_fraction_' + str(cf) + '_random_step_fraction_' + str(rsf))
		yaml.dump(config, open('without_ann/cross_fraction_' + str(cf) + '_random_step_fraction_' + str(rsf) + '/input.yaml', 'w'))