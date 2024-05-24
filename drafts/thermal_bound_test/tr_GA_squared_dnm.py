from dynamite import config
from dynamite.states import State
from dynamite.operators import zero, op_product, Operator
from dynamite.extras import majorana
from scipy.special import comb
from itertools import combinations
from collections import Counter
import numpy as np
import os
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--LA', type=int, required=True)
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()

config.shell=True
config.L = args.LA

# cache Majoranas to save time
NA = 2*args.LA
N = 2*NA
M = [majorana(idx) for idx in range(0, NA)]
RNG = np.random.default_rng(args.seed)
CPLS = RNG.normal(size=comb(N, 4, exact=True))  # the overall constant factor is 1/sqrt((N choose 4))
CPLS_MAP = {(i, j, k, l): CPLS[ind] for ind, (i, j, k, l) in enumerate(combinations(range(N), 4))}  # map from tuple to CPLS index}

def build_GA(A_inds: list[int]):
    """
    
    """
    GA = zero()
    # this is sorted, so the order of the indices is fine
    A_maj_inds = list(sorted([2*i for i in A_inds] + [2*i+1 for i in A_inds]))
    normalization_factor = 1.0 / comb(N, 4)
    iterator1 = list(combinations(A_maj_inds, 4))
    iterator2 = list(combinations(A_maj_inds, 4))

    for ind1 in iterator1:  # inds1 is a tuple of 4 indices
        for ind2 in iterator2:
            cdnl = len(set(ind1).union(set(ind2)))
            if cdnl == 4 or cdnl == 5 or cdnl == 7:
                continue
            C_cdnl = comb(N, NA) / comb(N-cdnl, NA-cdnl)
            ops = [M[i] for i in ind1] + [M[j] for j in ind2]
            GA += C_cdnl * CPLS_MAP[ind1] * CPLS_MAP[ind2] * op_product(ops)

    return normalization_factor * GA

def calc_tr_GA_squared(GA):
    """
    GA: array_like
    Calculate the trace of GA^2
    """
    GA_np = GA.to_numpy()
    return (GA_np @ GA_np).trace()

      
if __name__ == '__main__':
    filename = f'dnm_data_NA={2*args.LA}.csv'
    GA = build_GA(list(range(args.LA)))
    print(GA.to_numpy().shape)
    tr = calc_tr_GA_squared(GA)
    print(f'tr_GA_squared = {tr/(2**args.LA)/2**9}')
    # if not os.path.exists(filename):
    #     with open(filename, 'w') as f:
    #         f.write('NA,seed,tr_GA_squared\n')
    # with open (filename, 'a') as f:
    #     f.write(f'{2*args.LA}, {args.seed}, {tr/(2**args.LA)/2**9}\n')