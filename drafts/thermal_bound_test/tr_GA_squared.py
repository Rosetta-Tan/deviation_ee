# from dynamite import config
# from dynamite.states import State
# from dynamite.operators import zero, op_product
# from dynamite.extras import majorana
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

# config.shell=True
# config.L = args.LA

# cache Majoranas to save time
NA = 2*args.LA
N = 2*NA
# M = [majorana(idx) for idx in range(0, N)]
RNG = np.random.default_rng(args.seed)
CPLS = RNG.normal(size=comb(N, 4, exact=True))  # the overall constant factor is 1/sqrt((N choose 4))
CPLS_MAP = {(i, j, k, l): CPLS[ind] for ind, (i, j, k, l) in enumerate(combinations(range(N), 4))}  # map from tuple to CPLS index}

def inv(inds: list[int]) -> int:
    """
    Calculate the inverse number of a list of indices
    """
    inv_num = 0
    for i, ind1 in enumerate(inds):
        for ind2 in inds[i+1:]:
            if ind1 > ind2:
                inv_num += 1
    return inv_num
    

def calc_tr_GA_squared(A_inds: list[int]):
    """
    
    """
    tr = 0.
    ectd_ind_tuple = {}
    A_maj_inds = list(sorted([2*i for i in A_inds] + [2*i+1 for i in A_inds]))
    normalization_factor = 1.0 / comb(N, 4)
    iterator1 = list(combinations(A_maj_inds, 4))
    iterator2 = list(combinations(A_maj_inds, 4))

    for ind1 in iterator1:  # inds1 is a tuple of 4 indices
        # print(ind1)
        for ind2 in iterator2:
            cdnl = len(set(ind1).union(set(ind2)))
            if cdnl == 4 or cdnl == 5 or cdnl == 7:
                continue
            C_cdnl = comb(N, NA) / comb(N-cdnl, NA-cdnl)
            if cdnl == 6:
                ind12 = ind1+ ind2
                ind_counter = Counter(ind1 + ind2)
                ind12_repeat_once = [ind for ind in ind12 if ind_counter[ind] == 1]
                common_inds = [ind for ind in ind_counter if ind_counter[ind] == 2]
                ind1_repeat_once = [ind for ind in ind1 if ind_counter[ind] == 1]
                ind2_repeat_once = [ind for ind in ind2 if ind_counter[ind] == 1]
                
                ind1_reformed = ind1_repeat_once + common_inds
                ind2_reformed = common_inds + ind2_repeat_once

                total_inv = (inv(ind12_repeat_once) + 1 + inv(ind1) + inv(ind2) + inv(ind1_reformed) + inv(ind2_reformed)) % 2
                
                ind_repeat_once_sorted = tuple(sorted(ind12_repeat_once))
                ectd_ind_tuple[ind_repeat_once_sorted] = ectd_ind_tuple.get(ind_repeat_once_sorted, 0) + C_cdnl * CPLS_MAP[ind1] * CPLS_MAP[ind2] * (-1)**total_inv

                # if ind1 == (1, 2, 5, 7) and ind2 == (2, 4, 5, 6):
                #     print(f"""
                #         Counter: {ind_counter}
                #         ind1: {ind1}
                #         ind2: {ind2}
                #         ind12: {ind12}
                #         ind12_repeat_once: {ind12_repeat_once}
                #         common_inds: {common_inds}
                #         ind1_repeat_once: {ind1_repeat_once}
                #         ind2_repeat_once: {ind2_repeat_once}
                #         ind1_reformed: {ind1_reformed}
                #         ind2_reformed: {ind2_reformed}
                #         total_inv: {total_inv}
                #     """)

            else:
                total_inv = (inv(ind1 + ind2)) % 2
                # if(ind1 == (4,5,6,7) and ind2 == (0,1,2,3)):
                #     print(f'get here, { ectd_ind_tuple.get(tuple(sorted(ind1 + ind2)), 0)}')
                ectd_ind_tuple[tuple(sorted(ind1 + ind2))] = ectd_ind_tuple.get(tuple(sorted(ind1 + ind2)), 0) + C_cdnl * CPLS_MAP[ind1] * CPLS_MAP[ind2] * (-1)**total_inv
                # if (ind1 == (0,1,2,3) and ind2 == (4,5,6,7)) or (ind1 == (0,1,2,4) and ind2 == (3,5,6,7)):
                #     print(f"""
                #         ind1: {ind1}
                #         ind2: {ind2}
                #         total_inv: {total_inv}
                #         key: {tuple(sorted(ind1 + ind2))}
                #         ectd_ind_tuple.haskey: {tuple(sorted(ind1 + ind2)) in ectd_ind_tuple.keys()}
                #         ectd_ind_tuple.item: {ectd_ind_tuple[tuple(sorted(ind1 + ind2))]}
                #         K_J1: {CPLS_MAP[ind1]}
                #         K_J2: {CPLS_MAP[ind2]}
                #         C_cdnl: {C_cdnl}
                #     """)

    # with open ('debug.txt', 'w') as f:
    #     for key in ectd_ind_tuple.keys():
    #         f.write(f'{key}: {ectd_ind_tuple[key]}\n')

    for key in ectd_ind_tuple.keys():
        ectd_ind_tuple[key] *= normalization_factor
        tr += ectd_ind_tuple[key]**2 * 2**args.LA
    return tr

if __name__ == '__main__':
    tr = calc_tr_GA_squared(range(args.LA))
    print(tr/(2**args.LA)/2**9)
    if not os.path.exists(f'data_NA={2*args.LA}.csv'):
        with open (f'data_NA={2*args.LA}.csv', 'w') as f:
            f.write('NA,seed,tr_GA_squared\n')
    with open (f'data_NA={2*args.LA}.csv', 'a') as f:
        f.write(f'{2*args.LA}, {args.seed}, {tr/(2**args.LA)/2**9}\n')