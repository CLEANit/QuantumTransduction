#!/usr/bin/env python

import numpy as np
import dill


class GA:
    def __init__(self, models):
        self.models = models
