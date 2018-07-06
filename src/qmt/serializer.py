#!/usr/bin/env python

import dill
import subprocess
import coloredlogs, verboselogs


coloredlogs.install(level='INFO')
logger = verboselogs.VerboseLogger('QMT::serializer')


class Serializer:
    """
    Serialize the GA class so we can restart a calculation later.
    """
    def __init__(self, parser):
        subprocess.run(['mkdir -p restart'], shell=True)
        self.fname = 'restart/ga.dill'
        self.parser = parser

    def serialize(self, ga):
        """
        Serialize a GA class. The class is written to 'restart/ga.dill'

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
                if self.parser.getConfig() != ga.parser.getConfig():
                    logger.warning('The YAML configuration has changed, starting fresh...')
                    return None

                return ga
        except:
            return None
