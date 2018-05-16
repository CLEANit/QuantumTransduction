#!/usr/bin/env python

import dill

class Serializer:
    def __init__(self):
        self.fname = 'ga.dill'

    def serialize(self, ga):
        with open(self.fname, 'wb') as f:
            dill.dump(ga, f)

    def deserialize(self):
        try:
            with open(self.fname, 'rb') as f:
                ga = dill.load(f)
                return ga
        except:
            return None
