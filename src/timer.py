#!/usr/bin/env python

import time

class Timer:
    def __init__(self):
        self.start_time = None
        self.units = ['s', 'min', 'hrs', 'days']
        self.divs = [1., 60., 3600., 3600. * 24.]
        self.maxs = [60., 3600., 3600. * 24, 1000.]

    def start(self, val):
        self.start_time = val

    def stop(self, val, tformat='auto'):
        if tformat == 'auto':
            diff = val - self.start_time
            for u, d, m in zip(self.units, self.divs, self.maxs):
                if diff < m:
                    return '{:1.2f}'.format((val - self.start_time) / d) + ' ' + u
                else:
                    diff /= m


        # elif tformat == 's':
        #     return '{:1.2f} s'.format(time.clock() - self.start_time)
        # elif tformat == 'min':
        #     return '{:1.2f} s'.format((time.clock() - self.start_time) / 60. / n_threads)
        # elif tformat == 'hrs':
        #     return '{:1.2f} s'.format((time.clock() - self.start_time) / 3600. / n_threads)
        # elif tformat == 'days':
        #     return '{:1.2f} s'.format((time.clock() - self.start_time) / 3600. / 24. / n_threads)