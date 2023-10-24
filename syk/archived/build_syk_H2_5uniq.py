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
from dynamite.extras import majorana
from itertools import combinations
import numpy as np
from scipy.special import comb

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

# cache Majoranas to save time
M = [majorana(idx) for idx in range(0,N)]

et.begin_event('Solving coefficients')
def coeff(key):
    # assert len(key) == 2
    def reversed_num(lst):
        num = 0
        for i in range(len(lst)-1):
            for j in range(i+1,len(lst)):
               if lst[i] > lst[j]:
                   num += 1
        return num
    
    idxs_rem = [idx for idx in range(N) if idx not in key]
    idxs_com_all = list(combinations(idxs_rem, 3))  # 3 indices in common, pick all the possible 3 tuples
    key_divide = [(combo, tuple(set(key) - set(combo))) for combo in combinations(key, 1)]
    # combo: the index set for the first copy; set(key) - set(combo): the index set for the second copy
    coeff = 0.
    
    for idxs_com in idxs_com_all:
        for key1, key2 in key_divide:
            lst1 = list(key1) + list(idxs_com)
            lst2 = list(idxs_com) + list(key2)
            revserd_num_key_divide = reversed_num(list(key1) + list(key2))
            reversed_num_1 = reversed_num(lst1)
            reversed_num_2 = reversed_num(lst2)
            sgn = -1 if (revserd_num_key_divide + reversed_num_1 + reversed_num_2) % 2 == 0 else 1
            # sgn = 1 if (revserd_num_key_divide + reversed_num_1 + reversed_num_2) % 2 == 0 else -1
            '''
            Two index tuples have 3 indices in common.
            Insertion rule: Id = - chi_1 chi_2 chi_3 chi_1 chi_2 chi_3
            '''
            coeff += sgn*idxs_cpl[tuple(sorted(lst1))]*idxs_cpl[tuple(sorted(lst2))]
    return coeff

for j in range(n):
    et.begin_event('seed = '+str(j)+'coefficient dictionary')
    start = default_timer()

    # initialize random number generator
    # np.random.seed(j)
    rng = np.random.default_rng(seed=j)

    # prepare a single index tuple set, but used for two copies
    idxs_cpl = {}
    for idxs in combinations(range(N),4):
        idxs_cpl[idxs] = rng.normal()

    H2_5uniq = op_sum(coeff(key)*op_product((M[idx] for idx in key)) for key in list(combinations(range(N),2)))
    H2_5uniq.save(join(output_dir,f'H2_5uniq_{L}_{j}.msc'))

    # build the 5-element coefficient dictionary
    # maj_terms = {key: 0. for key in list(combinations(range(N),2))}
    # for key in maj_terms.keys():
    #     maj_terms[key] = coeff(key)
    # obj = {}
    # for key, val in maj_terms.items():
    #     obj[str(key)] = val
    # with open (join(output_dir,f'maj_terms_5uniq_{L}_{j}.json'),'w') as f:
    #     json.dump(obj,f,indent=4)
    
    et.end_event()
    print('Built Hamiltonian squared',j,'\t',timedelta(0,default_timer()-start))

et.begin_event('End\n\n')
print('Total max memory usage:',get_max_memory_usage()/(1E9),'Gb')