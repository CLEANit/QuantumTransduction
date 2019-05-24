#!/usr/bin/env python

import numpy as np
import cmocean.cm as cm
import sys
import subprocess

import progressbar as pb

bar = pb.ProgressBar()

files = subprocess.check_output('find . -name phase*', shell=True)

dir_to_file = {}

for elem in bar(files.split()):
    elem = elem.decode('utf-8')
    dir_to_file[elem.split('/')[1]] = elem.split()[-1]

print(dir_to_file)