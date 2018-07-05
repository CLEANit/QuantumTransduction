#!/usr/bin/env python

import dill
import subprocess

class Serializer:
    """
    Serialize the GA class so we can restart a calculation later.
    """
    def __init__(self):
        subprocess.run(['mkdir -p output'], shell=True)
        self.fname = 'output/ga.dill'

    def serialize(self, ga):
        """
        Serialize a GA class. The class is written to 'output/ga.dill'

        Parameters
        ----------
        ga : A GA class.
        """
        with open(self.fname, 'wb') as f:
            dill.dump(ga, f)

    def deserialize(self):
        """
        Try and load the serialized GA class.

        Returns
        -------
        If successful, a GA class. If not, returns None.
        """
        try:
            with open(self.fname, 'rb') as f:
                ga = dill.load(f)
                return ga
        except:
            return None
