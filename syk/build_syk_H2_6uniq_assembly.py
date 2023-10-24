from sys import argv
from math import sqrt
from os.path import join
import json

L = int(argv[1]) # number of spins (not # of majoranas!)
seed = int(argv[2]) # disorder realization sample number
output_dir = '/n/home01/ytan/scratch/deviation_ee/msc_syk'

from dynamite import config
config.L = L

from dynamite.operators import op_sum, op_product
from dynamite.tools import track_memory, get_max_memory_usage, get_cur_memory_usage
from dynamite.extras import majorana
from itertools import combinations
import numpy as np
from scipy.special import comb

from timeit import default_timer
from datetime import timedelta

track_memory()

class EventTimer:
    def __init__(self):
        self.global_start = default_timer()
        self.sub_start = None

    @classmethod
    def _td(cls,start,end):
        return timedelta(0,end-start)

    def end_event(self,t=None):

        if t is None:
            t = default_timer()

        print(' (%s, cur %s Gb, max %s Gb)' % (self._td(self.sub_start,t),
                                               str(get_cur_memory_usage()/1E9),
                                               str(get_max_memory_usage()/1E9)),sep='')
        ss = self.sub_start
        self.sub_start = None
        return t - ss

    def begin_event(self,name):
        new_start = default_timer()
        if self.sub_start is not None:
            d = self.end_event(new_start)
        else:
            d = None
        print(self._td(self.global_start,new_start),
              name,end='')
        self.sub_start = new_start
        return d

et = EventTimer()
et.begin_event('Start')
N = 2*L # number of Majoranas
J = 1.0

# cache Majoranas to save time
M = [majorana(idx) for idx in range(0,N)]

et.begin_event('seed = '+str(seed)+'coefficient dictionary')
start = default_timer()

# initialize random number generator
rng = np.random.default_rng(seed=seed)

# prepare the two different index tuple sets
idxs_cpl = {}
for idxs in combinations(range(N),4):
    idxs_cpl[idxs] = rng.normal()


et.end_event()
print('Built Hamiltonian squared',seed,'\t',timedelta(0,default_timer()-start))

et.begin_event('End\n\n')
print('Total max memory usage:',get_max_memory_usage()/(1E9),'Gb')