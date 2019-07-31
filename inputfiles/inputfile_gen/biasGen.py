#!/usr/bin/env python

import yaml
import os
import numpy as np
import sys

config = yaml.load(open(sys.argv[1], 'r'))


n_tests = 32

x = np.linspace(0.05, 0.5, n_tests)

for val in x:
	dirname = '%0.6f' % (val)
	os.mkdir('%0.6f' % (val))
	config['System']['bias'] = float(val)
	yaml.dump(config, open(dirname + '/input.yaml' , 'w'))