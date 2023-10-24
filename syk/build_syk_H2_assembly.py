from sys import argv
from math import sqrt
from os.path import join

L = int(argv[1]) # number of spins (not # of majoranas!)
seed = int(argv[2]) # disorder realization sample number
msc_dir = '/n/home01/ytan/scratch/deviation_ee/msc_syk'

from dynamite import config
config.L = L

from dynamite.operators import op_sum, op_product, Operator
from dynamite.extras import majorana
from itertools import combinations
import numpy as np
# from scipy.sparse.linalg import trace

from timeit import default_timer
from datetime import timedelta

from dynamite.tools import track_memory, get_max_memory_usage, get_cur_memory_usage
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

start = default_timer()

# load the cleverly decomposed H2 (without trace)
H2_6uniq = Operator.load(join(msc_dir,'H2_6uniq_'+str(L)+'_'+str(seed)+'.msc'))
H2_8uniq = Operator.load(join(msc_dir,'H2_8uniq_'+str(L)+'_'+str(seed)+'.msc'))
H2 = (J**2)*24/(N*(N-1)*(N-2)*(N-3))*op_sum([H2_6uniq, H2_8uniq])
H2_8uniq.save(join(msc_dir,f'H2_{L}_{seed}.msc'))

et.end_event()
print('Built Hamiltonian squared',seed,'\t',timedelta(0,default_timer()-start))

et.begin_event('End\n\n')
print('Total max memory usage:',get_max_memory_usage()/(1E9),'Gb')

