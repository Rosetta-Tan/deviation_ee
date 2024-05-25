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
from timeit import default_timer
import subprocess

class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[FlushStreamHandler()])

parser = argparse.ArgumentParser()
parser.add_argument('--LA', type=int, required=True)
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()

config.shell=False
config.L = args.LA

# cache Majoranas to save time
NA = 2*args.LA
N = 2*NA
M = [majorana(idx) for idx in range(0, NA)]
RNG = np.random.default_rng(args.seed)
CPLS = RNG.normal(size=comb(N, 4, exact=True))  # the overall constant factor is 1/sqrt((N choose 4))
CPLS_MAP = {(i, j, k, l): CPLS[ind] for ind, (i, j, k, l) in enumerate(combinations(range(N), 4))}  # map from tuple to CPLS index}

def build_GA(A_inds: list[int]):
    # copy global variables to local variables,
    # this will make the function faster
    local_NA = NA
    local_N = N
    local_M = M
    local_CPLS_MAP = CPLS_MAP

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
            C_cdnl = comb(local_N, local_NA) / comb(local_N-cdnl, local_NA-cdnl)
            ops = [local_M[i] for i in ind1] + [local_M[j] for j in ind2]
            GA += C_cdnl * local_CPLS_MAP[ind1] * local_CPLS_MAP[ind2] * op_product(ops)

    return normalization_factor * GA

def calc_tr_GA_squared(GA):
    """
    GA: array_like
    Calculate the trace of GA^2
    """
    GA_np = GA.to_numpy()
    return (GA_np @ GA_np).trace()
      
def streamline_tr_GA_squared():
    filename = f'dnm_data_NA={2*args.LA}.csv'
    start = default_timer()
    GA = build_GA(list(range(args.LA)))
    print(GA.to_numpy().shape)
    tr = calc_tr_GA_squared(GA)
    end = default_timer()
    print(f'time elapsed: {end-start}')
    print(f'tr_GA_squared = {tr/(2**args.LA)/2**9}')
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write('NA,seed,tr_GA_squared\n')
    with open (filename, 'a') as f:
        f.write(f'{2*args.LA}, {args.seed}, {tr/(2**args.LA)/2**9}\n')

if __name__ == '__main__':
    filename = f"Library/CloudStorage/GoogleDrive-yitan@g.harvard.edu/My Drive/from_cannon/deviation_ee/msc_GA/GA_LA={args.LA}_seed={args.seed}.msc"
    if not os.path.exists(filename):
        logging.info(f"Start building GA, LA={args.LA}, seed={args.seed}")
        GA = build_GA(list(range(args.LA)))
        GA.save(filename)
        logging.info(f"Finish building GA, LA={args.LA}, seed={args.seed}")
    else:
        logging.info(f"GA file exists, LA={args.LA}, seed={args.seed}")



