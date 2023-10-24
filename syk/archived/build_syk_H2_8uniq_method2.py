from sys import argv
from math import sqrt
from os.path import join
import json

L = int(argv[1]) # number of spins (not # of majoranas!)
n = int(argv[2]) # number of disorder samples
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

et.begin_event('Generating terms')
def gen_terms(key):
    assert len(key) == 8  # this is for the 8-unique element case
    def reversed_num(lst):
        num = 0
        for i in range(len(lst)-1):
            for j in range(i+1,len(lst)):
               if lst[i] > lst[j]:
                   num += 1
        return num
    key1_all = combinations(key, 4)  # choose 4 indices out of 8 to be in the first copy
    for key1 in key1_all:
        key2 = tuple(set(key) - set(key1))  # the remaining 2 indices are in the second copy
        reversed_num_1 = reversed_num(key1)
        reversed_num_2 = reversed_num(key2)
        sgn = 1 if (reversed_num_1 + reversed_num_2) % 2 == 0 else -1
        contribution = sgn*idxs_cpl[tuple(sorted(key1))]*idxs_cpl[tuple(sorted(key2))]
        yield contribution*op_product([M[key1[0]],M[key1[1]],M[key1[2]],M[key1[3]],M[key2[0]],M[key2[1]],M[key2[2]],M[key2[3]]])

for j in range(n):
    et.begin_event('seed = '+str(j)+'coefficient dictionary')
    start = default_timer()

    # initialize random number generator
    # np.random.seed(j)
    rng = np.random.default_rng(seed=j)

    # prepare the two different index tuple sets
    idxs_cpl = {}
    for idxs in combinations(range(N),4):
        idxs_cpl[idxs] = rng.normal()

    H2_8uniq_method2 = op_sum([op_sum(gen_terms(key)) for key in combinations(range(N),8)])
    H2_8uniq_method2.save(join(output_dir,f'H2_8uniq_method2_{L}_{j}.msc'))
    
    # build the 6-element coefficient dictionary
    # maj_terms = {key: 0. for key in list(combinations(range(N),4))}
    # for key in maj_terms.keys():
    #     maj_terms[key] = coeff(key)
    # obj = {}
    # obj1 = {}
    # for key, val in maj_terms.items():
    #     obj[str(key)] = val
    # with open(join(output_dir,f'maj_terms_6uniq_{L}_{j}.json'),'w') as f:
    #     json.dump(obj,f,indent=4)
    # for key, val in idxs_cpl.items():
    #     obj1[str(key)] = val
    # with open(join(output_dir,f'cpl_terms_6uniq_{L}_{j}.json'),'w') as f:
    #     json.dump(obj1,f,indent=4)

    et.end_event()
    print('Built Hamiltonian squared',j,'\t',timedelta(0,default_timer()-start))

et.begin_event('End\n\n')
print('Total max memory usage:',get_max_memory_usage()/(1E9),'Gb')