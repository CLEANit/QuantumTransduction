#!/usr/bin/env python

import time

class Timer:
    """
    A simple timer class which can be used like a stop watch. Used for timing different parts of the code.
    """

    def __init__(self):
        self.start_time = None
        self.units = ['s', 'min', 'hrs', 'days']
        self.divs = [1., 60., 3600., 3600. * 24.]
        self.maxs = [60., 3600., 3600. * 24, 1000.]

    def start(self):
        """
        Start the stop watch.
        """
        self.start_time = time.time()

    def stop(self, tformat='auto'):
        """
        Stop the stop watch.

        Parameters
        ----------
        tformat : Format of the time. Accepts: 's', 'min', 'hrs', 'days'. Default is 'auto', where it tries
        to figure out what the best format is for readability. 
        """
        if tformat == 'auto':
            diff = time.time() - self.start_time
            for u, d, m in zip(self.units, self.divs, self.maxs):
                if diff < m:
                    return '{:1.2f}'.format(diff) + ' ' + u
                else:
                    diff /= m
