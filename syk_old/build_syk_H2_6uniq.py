from sys import argv
import os
from math import sqrt
from dynamite import config
from dynamite.operators import op_sum, op_product
from dynamite.tools import track_memory, get_max_memory_usage, get_cur_memory_usage
from dynamite.extras import majorana
from itertools import combinations
import numpy as np
from scipy.special import comb
import argparse

from timeit import default_timer
from datetime import timedelta

parser = argparse.ArgumentParser()
parser.add_argument('-L', type=int, required=True, help='system size')
parser.add_argument('-J', type=float, required=False, default=1.0, help='coupling strength')
parser.add_argument('--seed', type=int, required=True, help='sample seed')
parser.add_argument('--dirc', type=str, required=False, default='/n/home01/ytan/scratch/deviation_ee/msc_syk',  help='output directory')
args = parser.parse_args()
L = args.L
J = args.J
seed = args.seed
output_dir = args.dirc
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

config.L = L
N = 2*L # number of Majoranas

# cache Majoranas to save time
M = [majorana(idx) for idx in range(0,N)]

def coeff(key):
    # assert len(key) == 4
    def reversed_num(lst):
        num = 0
        for i in range(len(lst)-1):
            for j in range(i+1,len(lst)):
               if lst[i] > lst[j]:
                   num += 1
        return num
    
    idxs_rem = [idx for idx in range(N) if idx not in key]
    idxs_com_all = list(combinations(idxs_rem, 2))  # 2 indices in common, pick all the possible tuples
    key_divide = [(combo, tuple(set(key) - set(combo))) for combo in combinations(key, 2)]
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
            Two index tuples have 2 indices in common.
            Insertion rule: Id = - chi_1 chi_2 chi_1 chi_2
            '''
            coeff += sgn*idxs_cpl[tuple(sorted(lst1))]*idxs_cpl[tuple(sorted(lst2))]
    return coeff

start = default_timer()

# initialize random number generator
# np.random.seed(j)
rng = np.random.default_rng(seed=seed)

# prepare the two different index tuple sets
idxs_cpl = {}
for idxs in combinations(range(N),4):
    idxs_cpl[idxs] = rng.normal()

H2_6uniq = op_sum(coeff(key)*op_product((M[idx] for idx in key)) for key in list(combinations(range(N),4)))
H2_6uniq.save(os.path.join(output_dir,f'H2_6uniq_L={L}_seed={seed}.msc'))


print(f'build 6uniq L={L} seed = {seed}; time elapsed: {timedelta(0,default_timer()-start)}')